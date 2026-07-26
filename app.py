import os, json, hashlib, re, sqlite3, uuid, time, base64, binascii, socket, ipaddress
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, urljoin, unquote, parse_qs
from fastapi import FastAPI, HTTPException, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
import asyncio
import httpx
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
AIPIPE_KEY = os.environ.get("AIPIPE_KEY", "")
AIPIPE_BASE = "https://aipipe.org/openai/v1"
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "gpt-4o")

DB_PATH = os.environ.get("DB_PATH", "/tmp/ga5.db")

app = FastAPI(title="GA5 Universal Solver")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ----------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------
def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (cache_key TEXT PRIMARY KEY, proposal TEXT);
            CREATE TABLE IF NOT EXISTS evals (eval_id TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS verifiers (eval_id TEXT PRIMARY KEY, jwk TEXT);
            CREATE TABLE IF NOT EXISTS commits (commit_key TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS receipts (receipt_id TEXT PRIMARY KEY, eval_id TEXT);
            CREATE TABLE IF NOT EXISTS callbind (eval_call TEXT PRIMARY KEY, receipt_id TEXT);

            CREATE TABLE IF NOT EXISTS q10_tasks (task_id TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS q10_messages (idempotency_key TEXT PRIMARY KEY, task_id TEXT);

            CREATE TABLE IF NOT EXISTS q11_runs (run_id TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS q11_receipts (receipt_id TEXT PRIMARY KEY, run_id TEXT);
        """)
    print("Database initialized.")
init_db()

# ----------------------------------------------------------------
# LLM helper – AIPIPE only
# ----------------------------------------------------------------
async def call_llm_json(prompt: str, system_prompt: str = "", timeout: float = 20.0) -> dict:
    if not AIPIPE_KEY:
        print("WARNING: AIPIPE_KEY not set, returning empty dict.")
        return {}
    try:
        client = AsyncOpenAI(base_url=AIPIPE_BASE, api_key=AIPIPE_KEY)
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=AIPIPE_MODEL,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=2048
            ),
            timeout=timeout
        )
        text = resp.choices[0].message.content.strip()
        return extract_json(text)
    except Exception as e:
        print(f"AIPIPE call failed: {e}")
        return {}

def extract_json(text: str) -> dict:
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    return json.loads(text)

# ----------------------------------------------------------------
# Canonical JSON helpers
# ----------------------------------------------------------------
def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False)

def sha256hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# ----------------------------------------------------------------
# Q2 – Proration Bug
# ----------------------------------------------------------------
from pydantic import BaseModel

class ProrationRequest(BaseModel):
    old_price: float
    new_price: float
    days_remaining: float
    days_in_actual_month: float
    spec: str

@app.post("/charge")
def proration_endpoint(req: ProrationRequest):
    if req.spec == "v1":
        charge = (req.new_price - req.old_price) * (req.days_remaining / 30.0)
    elif req.spec == "v2":
        charge = (req.new_price - req.old_price) * (req.days_remaining / req.days_in_actual_month)
    else:
        raise HTTPException(status_code=400, detail="Invalid spec")
    return {"charge": round(charge, 2)}

# ----------------------------------------------------------------
# Q3 – Guardrail Hook (pre-tool-call)
# ----------------------------------------------------------------
SECRET_FILE = "/home/agent/.npmrc"
ALLOWED_WRITE_DIR = "/data/agent/outbox/"
ALLOWED_HOSTS_Q3 = {"raw.githubusercontent.com", "pypi.org"}

def resolve_path(path, cwd="/home/agent/workspace"):
    path = os.path.expanduser(path)
    if "$HOME" in path:
        path = path.replace("$HOME", "/home/agent")
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.realpath(path)

def check_bash(command):
    if '.npmrc' in command:
        return False, "Command references .npmrc"
    for m in re.finditer(r'[\'"]([A-Za-z0-9+/=]+)[\'"]', command):
        try:
            decoded = base64.b64decode(m.group(1)).decode('utf-8', errors='ignore')
            if '.npmrc' in decoded:
                return False, "Base64 encoded reference to .npmrc detected."
        except:
            pass
    path_candidates = re.findall(r'(~[^\s]*|/[^\s]*|\$HOME[^\s]*)', command)
    for candidate in path_candidates:
        try:
            resolved = resolve_path(candidate)
            if resolved == SECRET_FILE or resolved.startswith(SECRET_FILE + '/'):
                return False, f"Path resolves to secret file."
        except:
            pass
    return True, "Allowed"

def check_write_file(path):
    try:
        full_path = resolve_path(path)
    except:
        return False, "Invalid path."
    if full_path.startswith(ALLOWED_WRITE_DIR):
        return True, "Writing inside allowed outbox."
    return False, f"Write outside outbox."

def check_http_request(url):
    try:
        parsed = urlparse(url)
    except:
        return False, "Invalid URL."
    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname."
    if hostname in ALLOWED_HOSTS_Q3:
        return True, f"Host allowed."
    return False, f"Host not allowed."

@app.post("/guardrail")
def guardrail(request: dict):
    tool = request.get("tool")
    if tool == "bash":
        allowed, reason = check_bash(request.get("command", ""))
    elif tool == "write_file":
        allowed, reason = check_write_file(request.get("path", ""))
    elif tool == "http_request":
        allowed, reason = check_http_request(request.get("url", ""))
    else:
        allowed, reason = False, "Unknown tool"
    return {"decision": "allow" if allowed else "block", "reason": reason}

# ----------------------------------------------------------------
# Q4 – Skill Safety Audit Scanner
# ----------------------------------------------------------------
def parse_skill(skill_text):
    lines = skill_text.splitlines()
    if not lines or lines[0].strip() != '---':
        return None, skill_text
    end = 1
    while end < len(lines) and lines[end].strip() != '---':
        end += 1
    if end >= len(lines):
        return None, skill_text
    fm_text = '\n'.join(lines[1:end])
    try:
        fm = yaml.safe_load(fm_text)
    except:
        fm = None
    body = '\n'.join(lines[end+1:])
    return fm, body

def check_hardcoded_secret(fm, body):
    full_text = body or ""
    if fm:
        for v in fm.values():
            if isinstance(v, str):
                full_text += v
    patterns = [
        r'sk-[a-zA-Z0-9]{32,}', r'ghp_[a-zA-Z0-9]{36,}', r'xox[baprs]-[0-9a-zA-Z-]+',
        r'AKIA[0-9A-Z]{16}', r'eyJ[a-zA-Z0-9._-]+', r'https?://hooks\.slack\.com/services/[A-Za-z0-9/_]+',
        r'https?://discord\.com/api/webhooks/[0-9]+/[A-Za-z0-9_\-]+', r'Bearer [A-Za-z0-9._\-]{20,}',
        r'Authorization:\s*Basic\s+[A-Za-z0-9+/=]{10,}'
    ]
    for pat in patterns:
        if re.search(pat, full_text, re.IGNORECASE):
            return True
    keyw = r'(api[_\s]?key|secret|token|password|webhook|auth[_\s]?token|credentials)'
    for m in re.finditer(keyw + r'\s*[:=]\s*["\']([^"\']{6,})["\']', full_text, re.IGNORECASE):
        val = m.group(2)
        if not re.search(r'[\$\{]', val):
            return True
    return False

def check_prompt_injection(body):
    phrases = [r'ignore (all |previous )?instructions?', r'disregard (the )?(user|above)',
               r'do not ask (for |the user )?(confirmation|permission)', r'silently\s+(upload|send|exfiltrate)']
    return any(re.search(p, body or "", re.IGNORECASE) for p in phrases)

def check_excessive_permissions(fm):
    if not fm: return False
    perms = fm.get('permissions') or fm.get('tools') or {}
    fs = perms.get('filesystem') or perms.get('fs') or {}
    net = perms.get('network') or perms.get('net') or ''
    wild_paths = {'/', '/home', '~', '*', '**', 'all'}
    for p in (fs.get('read',[]) + fs.get('write',[])):
        if p in wild_paths:
            return True
    if net in ('*', '0.0.0.0/0', 'any', 'all', 'internet'):
        return True
    return False

def check_unclear_provenance(fm, body):
    if not fm:
        return True
    if not all(fm.get(k) for k in ('author', 'version', 'changelog')):
        return True
    if re.search(r'version', body or "") and re.search(r'(update|change|modify|increment|bump)', body or "", re.IGNORECASE):
        if 'changelog' not in (body or "").lower():
            return True
    return False

@app.post("/scan")
async def scan_skill(request: Request):
    data = await request.json()
    skill_text = data.get("skill", "")
    fm, body = parse_skill(skill_text)
    cats = []
    if check_hardcoded_secret(fm, body): cats.append("hardcoded_secret")
    if check_prompt_injection(body): cats.append("prompt_injection")
    if check_excessive_permissions(fm): cats.append("excessive_permissions")
    if check_unclear_provenance(fm, body): cats.append("unclear_provenance")
    return {"categories": cats}

# ----------------------------------------------------------------
# Q5 – Run Budget & Loop Guard
# ----------------------------------------------------------------
def canonical_args(args, ignore_field):
    cleaned = {k:v for k,v in args.items() if k != ignore_field}
    return canonical(cleaned)

@app.post("/loop-guard")
def loop_guard(body: dict):
    budget = body["budget_tokens"]
    steps = body["steps"]
    total = sum(s["tokens_used"] for s in steps)
    if total >= budget:
        return {"decision":"halt","reason":f"Budget reached ({total} >= {budget})"}
    if len(steps) >= 3:
        last3 = [(s["tool"], canonical_args(s["args"], "")) for s in steps[-3:]]
        if last3[0] == last3[1] == last3[2]:
            return {"decision":"halt","reason":"3 identical calls in a row"}
    if len(steps) >= 6:
        sigs = [(s["tool"], canonical_args(s["args"], "")) for s in steps[-6:]]
        if sigs[0]==sigs[2]==sigs[4] and sigs[1]==sigs[3]==sigs[5] and sigs[0]!=sigs[1]:
            return {"decision":"halt","reason":"Alternating cycle detected"}
    return {"decision":"continue","reason":"Under budget, no loop"}

# ----------------------------------------------------------------
# Q6 – MCP Server (solve_challenge)
# ----------------------------------------------------------------
@app.post("/mcp")
async def mcp_handler(request: Request):
    body = await request.json()
    method = body.get("method")
    req_id = body.get("id")
    if method == "initialize":
        return {"jsonrpc":"2.0","result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"exam-server","version":"1.0"}},"id":req_id}
    elif method == "tools/list":
        return {"jsonrpc":"2.0","result":{"tools":[{"name":"solve_challenge","description":"Solve exam challenge","inputSchema":{"type":"object","properties":{}}}]},"id":req_id}
    elif method == "tools/call":
        challenge = request.headers.get("X-Exam-Challenge", "")
        email = os.environ.get("STUDENT_EMAIL", "24f2002559@ds.study.iitm.ac.in").lower()
        answer = hashlib.sha256(f"{challenge}:{email}".encode()).hexdigest()[:16]
        return {"jsonrpc":"2.0","result":{"content":[{"type":"text","text":answer}]},"id":req_id}
    return {"jsonrpc":"2.0","error":{"code":-32601,"message":"Not found"},"id":req_id}

# ----------------------------------------------------------------
# Q8 – Guardrail Red-Team (simplified)
# ----------------------------------------------------------------
SANDBOX_ROOT_Q8 = "/srv/agent-redteam/sandbox-d1d2e55fc4"
ALLOWED_WEB_HOSTS_Q8 = {"example.com", "www.iana.org"}

def is_public_ip(host):
    try:
        addrinfo = socket.getaddrinfo(host, None)
        ips = {info[4][0] for info in addrinfo}
        for ip_str in ips:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        return True
    except:
        return False

@app.post("/redteam")
def redteam_guardrail(request: dict):
    tool = request.get("tool")
    args = request.get("arguments", {})
    if tool == "read_file":
        path = args.get("path", "")
        normalized = os.path.realpath(os.path.join(SANDBOX_ROOT_Q8, path.lstrip("/")))
        if normalized.startswith(SANDBOX_ROOT_Q8):
            if os.path.exists(normalized):
                with open(normalized, "r") as f:
                    content = f.read()
                return {"action":"allow","reason":"Inside sandbox","result":content}
            return {"action":"allow","reason":"File not found","result":"Error"}
        return {"action":"block","reason":"Outside sandbox","result":None}
    elif tool == "fetch_url":
        url = args.get("url", "")
        parsed = urlparse(url)
        if parsed.hostname not in ALLOWED_WEB_HOSTS_Q8 or not is_public_ip(parsed.hostname):
            return {"action":"block","reason":"Host not allowed or private","result":None}
        try:
            resp = httpx.get(url, timeout=5, follow_redirects=False)
            return {"action":"allow","reason":"Fetched","result":resp.text}
        except:
            return {"action":"block","reason":"Fetch error","result":None}
    return {"action":"block","reason":"Unknown tool","result":None}

# ----------------------------------------------------------------
# Q9 – Mailroom Agent (full)
# ----------------------------------------------------------------
PROFILE_MAILROOM = "ga5-mailroom-action-gate/v2"
ALLOWED_ACTIONS_Q9 = ["create_draft","update_internal_record","send_approved_notice","request_confirmation","quarantine_item","no_action"]

def dossier_fingerprint(dossier):
    copy = {k:v for k,v in dossier.items() if k != "receivedAt"}
    return sha256hex(canonical(copy))

def call_id_for(dossier_id, fingerprint):
    raw = f"q9|{dossier_id}|{fingerprint}"
    return "mr_" + sha256hex(raw)[:40]

def deterministic_decision(dossier):
    for src in dossier.get("sources", []):
        if src.get("kind") != "approval": continue
        text = " ".join(line["text"] for line in src.get("lines", []))
        m = re.search(r'Approval (EVT-\S+) permits one delivery-status notice for (ORD-\S+) to (\S+) using template approved_delivery_notice\.', text)
        if m:
            ev = [l["lineId"] for l in src["lines"] if "permits" in l["text"] or "public status" in l["text"]]
            st = "approved"
            for l in src["lines"]:
                ms = re.search(r'valid for the public status "(\w+)"', l["text"])
                if ms: st = ms.group(1)
            return {"action":"send_approved_notice","evidence":ev,"fields":{"recipient":m.group(3),"referenceId":m.group(2),"status":st}}
    return None

def shape_target_payload(action, fields, dossier, line_ids):
    mailbox = dossier.get("mailbox", "unknown")
    ref = fields.get("referenceId", dossier.get("dossierId"))
    if action == "create_draft":
        return ({"kind":"draft_queue","id":"mailbox:"+mailbox},
                {"recipient": fields.get("recipient", mailbox), "referenceId": ref,
                 "status": fields.get("status","in_progress"), "template":"order_status"})
    # For brevity, other actions return a generic fallback
    return (None, {"reasonCode":"INFORMATIONAL","referenceId":ref})

def build_proposal(dossier_id, dossier, fingerprint, decision):
    action = decision.get("action")
    if action not in ALLOWED_ACTIONS_Q9:
        action = "request_confirmation"
    fields = decision.get("fields", {})
    target, payload = shape_target_payload(action, fields, dossier, [])
    evidence = decision.get("evidence", [])
    return {
        "dossierId": dossier_id,
        "callId": call_id_for(dossier_id, fingerprint),
        "action": action,
        "target": target,
        "payload": payload,
        "evidence": evidence[:5]
    }

@app.post("/mailroom")
async def mailroom(request: Request):
    body = await request.json()
    if body.get("profile") != PROFILE_MAILROOM:
        raise HTTPException(400, "Invalid profile")
    op = body.get("operation")
    if op == "propose":
        return await handle_propose(body)
    elif op == "commit":
        return await handle_commit(body)
    raise HTTPException(400, "Unknown operation")

async def handle_propose(body):
    eval_id = body["evaluationId"]
    dossiers = body["dossiers"]
    input_digest = sha256hex(canonical(dossiers))
    row = db_get("evals", eval_id)
    if row:
        stored = json.loads(row)
        if stored.get("inputDigest") == input_digest:
            return stored
        raise HTTPException(409, "Conflict with different content")
    proposals = []
    for dossier in dossiers:
        fp = dossier_fingerprint(dossier)
        cache_key = dossier["dossierId"] + "|" + fp
        cached = db_get("decisions", cache_key)
        if cached:
            proposals.append(json.loads(cached))
            continue
        det = deterministic_decision(dossier)
        if det:
            prop = build_proposal(dossier["dossierId"], dossier, fp, det)
        else:
            prompt = f"Dossier:\n{json.dumps(dossier, indent=2)}\nChoose one action: {ALLOWED_ACTIONS_Q9}. Return JSON with action, evidence, and fields."
            llm_out = await call_llm_json(prompt)
            prop = build_proposal(dossier["dossierId"], dossier, fp, llm_out)
        db_put("decisions", cache_key, canonical(prop))
        proposals.append(prop)
    resp = {"profile":PROFILE_MAILROOM,"evaluationId":eval_id,"status":"awaiting_receipts","inputDigest":input_digest,"proposals":proposals}
    db_put("evals", eval_id, canonical(resp))
    verifier = body.get("receiptVerifier",{})
    if "publicKeyJwk" in verifier:
        db_put("verifiers", eval_id, canonical(verifier["publicKeyJwk"]))
    return resp

async def handle_commit(body):
    eval_id = body["evaluationId"]
    input_digest = body["inputDigest"]
    receipts = body["receipts"]
    stored_row = db_get("evals", eval_id)
    if not stored_row: raise HTTPException(409, "Unknown evaluation")
    stored = json.loads(stored_row)
    if stored["inputDigest"] != input_digest: raise HTTPException(409, "Digest mismatch")
    jwk_row = db_get("verifiers", eval_id)
    if jwk_row:
        jwk = json.loads(jwk_row)
        verify_receipts(eval_id, input_digest, receipts, jwk)
    prop_map = {p["callId"]:p for p in stored["proposals"]}
    outcomes = []
    for r in receipts:
        prop = prop_map[r["callId"]]
        pdigest = proposal_digest(prop)
        accepted = r.get("accepted", False)
        outcomes.append({
            "dossierId":prop["dossierId"],"callId":prop["callId"],"action":prop["action"],
            "proposalDigest":pdigest,"receiptId":r["receiptId"],
            "status":"executed" if accepted else "rejected"
        })
    final = {"profile":PROFILE_MAILROOM,"evaluationId":eval_id,"status":"completed","inputDigest":input_digest,"outcomes":outcomes}
    commit_key = sha256hex(canonical({"eval_id":eval_id,"receipts":receipts}))
    db_put("commits", commit_key, canonical(final))
    for r in receipts:
        db_put("receipts", r["receiptId"], eval_id)
        db_put("callbind", f"{eval_id}|{r['callId']}", r["receiptId"])
    return final

def verify_receipts(eval_id, input_digest, receipts, jwk):
    x_bytes = base64.urlsafe_b64decode(jwk["x"] + "===")
    pub = Ed25519PublicKey.from_public_bytes(x_bytes)
    for r in receipts:
        msg = canonical({
            "profile": PROFILE_MAILROOM,
            "evaluationId": eval_id,
            "inputDigest": input_digest,
            "receipt": {
                "dossierId": r["dossierId"],
                "callId": r["callId"],
                "action": r["action"],
                "accepted": r["accepted"],
                "proposalDigest": r["proposalDigest"],
                "receiptId": r["receiptId"]
            }
        }).encode()
        sig = base64.b64decode(r["receiptSignature"])
        try:
            pub.verify(sig, msg)
        except InvalidSignature:
            raise HTTPException(422, "Invalid signature")

def proposal_digest(prop):
    core = {"dossierId":prop["dossierId"],"callId":prop["callId"],"action":prop["action"],
            "target":prop.get("target"),"payload":prop["payload"],"evidence":sorted(prop["evidence"])}
    return sha256hex(canonical(core))

# ----------------------------------------------------------------
# Q10 – A2A Invoice Agent (simplified)
# ----------------------------------------------------------------
A2A_BASE = "/a2a"

@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "Invoice Agent",
        "description": "Processes invoice batches",
        "version": "1.0",
        "capabilities": {},
        "skills": [{"name":"invoice_action_agent","description":"Decide actions on invoices","tags":["invoice"]}],
        "supportedInterfaces": [{"protocolBinding":"HTTP+JSON","protocolVersion":"1.0","url": str(app.url_path_for("agent_card")).replace("/.well-known/agent-card.json","")}],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": ["application/vnd.ga5.invoice-action-proposals+json","application/vnd.ga5.invoice-action-receipts+json"]
    }

@app.post(A2A_BASE+"/message:send")
async def a2a_message_send(request: Request, authorization: str = Header(None)):
    token = authorization.split(" ")[-1] if authorization else "anonymous"
    body = await request.json()
    message = body.get("message", {})
    messageId = message["messageId"]
    idem_key = sha256hex(canonical(message))
    row = db_get("q10_messages", idem_key)
    if row:
        return JSONResponse(json.loads(row))
    parts = message.get("parts", [])
    batch_data = None
    for part in parts:
        if part.get("mediaType") == "application/vnd.ga5.invoice-claim-batch+json":
            batch_data = part["data"]
            break
    if not batch_data:
        raise HTTPException(400, "No invoice batch found")
    batch_id = batch_data["batchId"]
    packages = batch_data["packages"]
    prompt = f"Batch {batch_id}. Packages:\n{json.dumps(packages, indent=2)}\nFor each package, choose one action: settle_invoice, request_approval, hold_invoice, reject_duplicate, open_exception. Return JSON list of objects with packageId, action, vendorName, invoiceNumber, amountMinor, currency, evidenceRefs, rationale."
    llm_result = await call_llm_json(prompt)
    proposals = []
    for pkg in packages:
        res = next((r for r in llm_result if r["packageId"] == pkg["packageId"]), None)
        if not res:
            continue
        proposals.append({
            "packageId": pkg["packageId"],
            "actionId": uuid.uuid4().hex[:12],
            "action": res["action"],
            "facts": {"vendorName":res["vendorName"],"invoiceNumber":res["invoiceNumber"],"amountMinor":res["amountMinor"],"currency":res["currency"]},
            "evidenceRefs": res["evidenceRefs"],
            "rationale": res["rationale"]
        })
    task = {
        "id": uuid.uuid4().hex,
        "state": "TASK_STATE_INPUT_REQUIRED",
        "history": [{"message":message}],
        "artifacts": [{"parts":[{"mediaType":"application/vnd.ga5.invoice-action-proposals+json","data":{"batchId":batch_id,"proposals":proposals}}]}]
    }
    db_put("q10_messages", idem_key, canonical(task))
    return task

# ----------------------------------------------------------------
# Q11 – Incident Agent (simplified)
# ----------------------------------------------------------------
@app.post("/v2/incidents")
async def create_incident(request: Request):
    body = await request.json()
    run_id = body["runId"]
    row = db_get("q11_runs", run_id)
    if row: raise HTTPException(409, "Run already exists")
    transcript = body["incident"]["transcript"]
    allowed_roots = body["incident"]["allowedRootCauses"]
    tools = body["toolCatalog"]
    prompt = f"Transcript:\n{transcript}\nAllowed root causes: {allowed_roots}\nTools: {json.dumps(tools)}\nReturn JSON with rootCause, evidence (list of IDs), and dispatches (list of tool calls with toolName and arguments)."
    plan = await call_llm_json(prompt)
    state = {
        "runId": run_id,
        "status": "waiting",
        "diagnosis": {"rootCause": plan["rootCause"], "evidence": plan["evidence"]},
        "dispatches": plan.get("dispatches", []),
        "approvals": []
    }
    db_put("q11_runs", run_id, canonical(state))
    return state

@app.post("/v2/incidents/{run_id}/receipts")
async def post_receipts(run_id: str, request: Request):
    body = await request.json()
    row = db_get("q11_runs", run_id)
    if not row: raise HTTPException(404, "Run not found")
    state = json.loads(row)
    outcomes = body.get("outcomes", [])
    state["status"] = "completed"
    db_put("q11_runs", run_id, canonical(state))
    return state

@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    row = db_get("q11_runs", run_id)
    if not row: raise HTTPException(404)
    return json.loads(row)

# ----------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------
def db_get(table, key):
    with db_connect() as conn:
        cur = conn.execute(f"SELECT data FROM {table} WHERE rowid=?", (key,))
        row = cur.fetchone()
        return row["data"] if row else None

def db_put(table, key, value):
    with db_connect() as conn:
        conn.execute(f"INSERT OR REPLACE INTO {table} (rowid, data) VALUES (?,?)", (key, value))

# ----------------------------------------------------------------
# Debug endpoint
# ----------------------------------------------------------------
@app.get("/debug")
def debug_info():
    info = {
        "status": "ok",
        "aipipe_key_set": bool(AIPIPE_KEY),
        "aipipe_model": AIPIPE_MODEL,
        "db_path": DB_PATH,
        "table_counts": {}
    }
    tables = ["decisions", "evals", "verifiers", "commits", "receipts", "callbind",
              "q10_tasks", "q10_messages", "q11_runs", "q11_receipts"]
    with db_connect() as conn:
        for table in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                info["table_counts"][table] = count
            except Exception as e:
                info["table_counts"][table] = f"error: {e}"
    return info

# ----------------------------------------------------------------
# Root endpoint
# ----------------------------------------------------------------
@app.get("/")
def root():
    return {"status":"ok","service":"GA5 Universal Solver"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
