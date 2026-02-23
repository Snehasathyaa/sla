"""
Smart SLA Prediction & Escalation System — Flask Application
Kerala Government Services | Uses built-in sqlite3
"""

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, session, jsonify, g)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
import sqlite3
import joblib
import numpy as np
import os
import json
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kerala-sla-secret-2024'
DATABASE = 'sla_system.db'

# ── ML Models ─────────────────────────────────────────────────────────────────
clf = reg = encoders = None

def load_models():
    global clf, reg, encoders
    try:
        clf = joblib.load('models/sla_classifier.pkl')
        reg = joblib.load('models/resolution_regressor.pkl')
        encoders = joblib.load('models/encoders.pkl')
        print("✔ ML models loaded")
    except Exception as e:
        print(f"⚠ Models not loaded ({e}). Run train_model.py first.")

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORIES = ['Housing', 'Education', 'Healthcare', 'Sanitation',
              'Revenue', 'Public Works', 'Social Security', 'Water Supply']
DEPARTMENTS = {
    'Housing': 'Housing Board', 'Education': 'Education Department',
    'Healthcare': 'Health Department', 'Sanitation': 'Municipality',
    'Revenue': 'Revenue Department', 'Public Works': 'PWD',
    'Social Security': 'Social Welfare', 'Water Supply': 'Kerala Water Authority'
}
PRIORITIES = ['Low', 'Medium', 'High', 'Critical']
SLA_THRESHOLDS = {'Low': 15, 'Medium': 10, 'High': 5, 'Critical': 2}
DISTRICTS = ['Thiruvananthapuram','Kollam','Pathanamthitta','Alappuzha',
             'Kottayam','Idukki','Ernakulam','Thrissur','Palakkad',
             'Malappuram','Kozhikode','Wayanad','Kannur','Kasaragod']
CATEGORY_MULTIPLIER = {
    'Housing':1.4,'Education':1.0,'Healthcare':0.85,'Sanitation':0.9,
    'Revenue':1.2,'Public Works':1.5,'Social Security':1.1,'Water Supply':0.9
}

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    return (rv[0] if rv else None) if one else rv

def modify_db(query, args=()):
    db = get_db()
    cur = db.execute(query, args)
    db.commit()
    return cur.lastrowid

def init_db():
    db = sqlite3.connect(DATABASE)
    db.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'citizen',
        district TEXT,
        department TEXT,
        experience_years INTEGER DEFAULT 5,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id TEXT UNIQUE,
        user_id INTEGER,
        assigned_to INTEGER,
        category TEXT, department TEXT, priority TEXT,
        district TEXT, description TEXT,
        status TEXT DEFAULT 'Pending',
        sla_threshold INTEGER,
        predicted_days REAL,
        sla_risk_score REAL,
        sla_risk_label TEXT,
        submitted_at TEXT DEFAULT (datetime('now')),
        resolved_at TEXT,
        escalated INTEGER DEFAULT 0,
        escalated_at TEXT,
        resubmission INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id INTEGER,
        action TEXT,
        performed_by INTEGER,
        timestamp TEXT DEFAULT (datetime('now')),
        notes TEXT
    );
    ''')
    db.commit()
    db.close()

# ── Complaint wrapper ──────────────────────────────────────────────────────────
class C:
    def __init__(self, row):
        for k in row.keys():
            setattr(self, k, row[k])
        self.escalated = bool(getattr(self,'escalated',0))
        for attr in ('submitted_at','resolved_at','escalated_at'):
            val = getattr(self, attr, None)
            setattr(self, attr, _parse_dt(val))

    @property
    def days_pending(self):
        end = self.resolved_at or datetime.utcnow()
        return max(0,(end - self.submitted_at).days) if self.submitted_at else 0
    @property
    def days_remaining(self):
        if not self.submitted_at: return self.sla_threshold
        return (self.submitted_at + timedelta(days=self.sla_threshold) - datetime.utcnow()).days
    @property
    def sla_violated(self):
        return self.days_pending > self.sla_threshold and self.status not in ('Resolved','Closed')
    @property
    def progress_pct(self):
        return min(100, int(self.days_pending / max(1, self.sla_threshold) * 100))

def _parse_dt(s):
    if not s: return None
    for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%d %H:%M:%S.%f'):
        try: return datetime.strptime(s, fmt)
        except: pass
    return None

def wrap(rows, one=False):
    if one: return C(rows) if rows else None
    return [C(r) for r in rows]

# ══════════════════════════════════════════════════════════════════════════════
# ML
# ══════════════════════════════════════════════════════════════════════════════

def predict_sla(category, department, priority, district, description,
                resubmission=False, officer_exp=10, workload=30):
    sla_thresh = SLA_THRESHOLDS[priority]
    multiplier = CATEGORY_MULTIPLIER.get(category, 1.0)
    if clf and encoders:
        try:
            feat = np.array([[
                encoders['category'].transform([category])[0],
                encoders['department'].transform([department])[0],
                encoders['priority'].transform([priority])[0],
                encoders['district'].transform([district])[0],
                1, datetime.now().weekday(),
                officer_exp, workload, len(description),
                int(resubmission), sla_thresh
            ]])
            risk_prob = clf.predict_proba(feat)[0][1]
            pred_days = float(reg.predict(feat)[0])
        except:
            risk_prob = _fb_risk(priority, multiplier, workload)
            pred_days = sla_thresh * multiplier
    else:
        risk_prob = _fb_risk(priority, multiplier, workload)
        pred_days = sla_thresh * multiplier
    score = round(risk_prob*100, 1)
    label = 'Critical' if score>=75 else 'High' if score>=50 else 'Medium' if score>=25 else 'Low'
    return score, label, round(pred_days, 1)

def _fb_risk(priority, multiplier, workload):
    base = {'Low':0.15,'Medium':0.35,'High':0.60,'Critical':0.80}[priority]
    return min(0.98, base * multiplier * (1 + workload/200))

def log_action(cid, action, uid, notes=''):
    with sqlite3.connect(DATABASE) as db:
        db.execute('INSERT INTO audit_logs (complaint_id,action,performed_by,notes) VALUES (?,?,?,?)',
                   [cid, action, uid, notes])

def gen_cid():
    db = sqlite3.connect(DATABASE)
    count = db.execute('SELECT COUNT(*) FROM complaints').fetchone()[0] + 1
    db.close()
    return f'KL{datetime.now().year}-{str(count).zfill(5)}'

# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if 'user_id' not in session:
            flash('Please log in first.','warning')
            return redirect(url_for('login'))
        return f(*a, **kw)
    return dec

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def dec(*a, **kw):
            if 'user_id' not in session: return redirect(url_for('login'))
            if session.get('user_role') not in roles:
                flash('Access denied.','danger')
                return redirect(url_for('index'))
            return f(*a, **kw)
        return dec
    return decorator

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    stats = {
        'total':   query_db('SELECT COUNT(*) FROM complaints',one=True)[0],
        'pending': query_db("SELECT COUNT(*) FROM complaints WHERE status='Pending'",one=True)[0],
        'resolved':query_db("SELECT COUNT(*) FROM complaints WHERE status IN ('Resolved','Closed')",one=True)[0],
        'escalated':query_db('SELECT COUNT(*) FROM complaints WHERE escalated=1',one=True)[0],
    }
    recent = wrap(query_db('SELECT * FROM complaints ORDER BY submitted_at DESC LIMIT 6'))
    return render_template('index.html', stats=stats, recent=recent)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip()
        pw    = request.form.get('password','')
        user  = query_db('SELECT * FROM users WHERE email=?',[email],one=True)
        if user and check_password_hash(user['password'], pw):
            session.update({'user_id':user['id'],'user_name':user['name'],'user_role':user['role']})
            flash(f"Welcome back, {user['name']}!",'success')
            dest = {'admin':'admin_dashboard','official':'official_dashboard'}.get(user['role'],'citizen_dashboard')
            return redirect(url_for(dest))
        flash('Invalid credentials.','danger')
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name, email, pw = (request.form.get(k,'').strip() for k in ('name','email','password'))
        district = request.form.get('district','')
        role     = request.form.get('role','citizen')
        if query_db('SELECT id FROM users WHERE email=?',[email],one=True):
            flash('Email already registered.','danger')
            return redirect(url_for('register'))
        modify_db('INSERT INTO users (name,email,password,role,district) VALUES (?,?,?,?,?)',
                  [name,email,generate_password_hash(pw),role,district])
        flash('Account created! Please log in.','success')
        return redirect(url_for('login'))
    return render_template('register.html', districts=DISTRICTS)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.','info')
    return redirect(url_for('index'))

@app.route('/track', methods=['GET','POST'])
def track():
    complaint = None
    if request.method == 'POST':
        cid = request.form.get('complaint_id','').strip().upper()
        row = query_db('SELECT * FROM complaints WHERE complaint_id=?',[cid],one=True)
        complaint = wrap(row, one=True) if row else None
        if not complaint: flash('Complaint not found.','warning')
    return render_template('track.html', complaint=complaint)

# ── Citizen ──────────────────────────────────────────────────────────────────

@app.route('/citizen/dashboard')
@login_required
def citizen_dashboard():
    user = query_db('SELECT * FROM users WHERE id=?',[session['user_id']],one=True)
    complaints = wrap(query_db('SELECT * FROM complaints WHERE user_id=? ORDER BY submitted_at DESC',[session['user_id']]))
    return render_template('citizen/dashboard.html', user=user, complaints=complaints)

@app.route('/citizen/submit', methods=['GET','POST'])
@login_required
def submit_complaint():
    if request.method == 'POST':
        cat  = request.form.get('category')
        pri  = request.form.get('priority')
        dist = request.form.get('district')
        desc = request.form.get('description','')
        resub = request.form.get('resubmission')=='yes'
        dept  = DEPARTMENTS.get(cat,'General')
        sla   = SLA_THRESHOLDS[pri]
        officer = query_db('SELECT * FROM users WHERE role=? AND department=?',['official',dept],one=True)
        exp  = officer['experience_years'] if officer else 5
        wl   = query_db("SELECT COUNT(*) FROM complaints WHERE assigned_to=? AND status='In Progress'",
                        [officer['id'] if officer else -1],one=True)[0] if officer else 20
        risk,label,days = predict_sla(cat,dept,pri,dist,desc,resub,exp,wl)
        cid   = gen_cid()
        now   = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        esc   = 1 if pri=='Critical' and risk>=75 else 0
        stat  = 'Escalated' if esc else 'Pending'
        new_id = modify_db(
            '''INSERT INTO complaints (complaint_id,user_id,assigned_to,category,department,priority,
               district,description,sla_threshold,predicted_days,sla_risk_score,sla_risk_label,
               status,escalated,escalated_at,resubmission,submitted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            [cid,session['user_id'],officer['id'] if officer else None,
             cat,dept,pri,dist,desc,sla,days,risk,label,stat,esc,now if esc else None,int(resub),now])
        log_action(new_id,'Complaint submitted',session['user_id'],f'{cat},{pri}')
        flash(f'Complaint submitted! ID: {cid} | Risk: {label} ({risk}%)','success')
        return redirect(url_for('citizen_dashboard'))
    return render_template('citizen/submit.html',
                           categories=CATEGORIES,priorities=PRIORITIES,districts=DISTRICTS)

# ── Official ─────────────────────────────────────────────────────────────────

@app.route('/official/dashboard')
@login_required
@role_required('official','admin')
def official_dashboard():
    user = query_db('SELECT * FROM users WHERE id=?',[session['user_id']],one=True)
    q = 'SELECT * FROM complaints ORDER BY submitted_at DESC' if user['role']=='admin' \
        else 'SELECT * FROM complaints WHERE assigned_to=? ORDER BY submitted_at DESC'
    args = [] if user['role']=='admin' else [session['user_id']]
    complaints = wrap(query_db(q,args))
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    for c in complaints:
        if c.status not in ('Resolved','Closed','Escalated') and c.sla_violated and not c.escalated:
            modify_db("UPDATE complaints SET escalated=1,escalated_at=?,status='Escalated' WHERE id=?",[now,c.id])
            log_action(c.id,'Auto-escalated',session['user_id'])
    complaints = wrap(query_db(q,args))
    stats = {'total':len(complaints),
             'pending':sum(1 for c in complaints if c.status=='Pending'),
             'in_progress':sum(1 for c in complaints if c.status=='In Progress'),
             'escalated':sum(1 for c in complaints if c.escalated),
             'resolved':sum(1 for c in complaints if c.status in ('Resolved','Closed'))}
    return render_template('official/dashboard.html',user=user,complaints=complaints,stats=stats)

@app.route('/official/update/<int:cid>', methods=['POST'])
@login_required
@role_required('official','admin')
def update_complaint(cid):
    new_status = request.form.get('status')
    notes = request.form.get('notes','')
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    resolved_at = now if new_status in ('Resolved','Closed') else None
    esc = 1 if new_status=='Escalated' else 0
    esc_at = now if new_status=='Escalated' else None
    modify_db('UPDATE complaints SET status=?,resolved_at=COALESCE(?,resolved_at),escalated=COALESCE(?,escalated),escalated_at=COALESCE(?,escalated_at) WHERE id=?',
              [new_status,resolved_at,esc if esc else None,esc_at,cid])
    log_action(cid,f'Status→{new_status}',session['user_id'],notes)
    flash(f'Updated to {new_status}.','success')
    return redirect(request.referrer or url_for('official_dashboard'))

# ── Admin ─────────────────────────────────────────────────────────────────────

@app.route('/admin/dashboard')
@login_required
@role_required('admin')
def admin_dashboard():
    user = query_db('SELECT * FROM users WHERE id=?',[session['user_id']],one=True)
    complaints = wrap(query_db('SELECT * FROM complaints ORDER BY submitted_at DESC'))
    total = len(complaints)
    violated = sum(1 for c in complaints if c.sla_violated)
    stats = {'total':total,
             'pending':sum(1 for c in complaints if c.status=='Pending'),
             'in_progress':sum(1 for c in complaints if c.status=='In Progress'),
             'escalated':sum(1 for c in complaints if c.escalated),
             'resolved':sum(1 for c in complaints if c.status in ('Resolved','Closed')),
             'violated':violated,
             'violation_rate':round(violated/total*100 if total else 0,1),
             'officials':query_db("SELECT COUNT(*) FROM users WHERE role='official'",one=True)[0],
             'citizens':query_db("SELECT COUNT(*) FROM users WHERE role='citizen'",one=True)[0]}
    cat_data = {}
    for c in complaints:
        cat_data.setdefault(c.category,{'total':0,'violated':0})
        cat_data[c.category]['total'] += 1
        if c.sla_violated: cat_data[c.category]['violated'] += 1
    audit_logs = query_db('''SELECT a.*,u.name as uname FROM audit_logs a
                             LEFT JOIN users u ON a.performed_by=u.id
                             ORDER BY a.timestamp DESC LIMIT 20''')
    graphs = sorted(f for f in os.listdir('static/graphs') if f.endswith('.png')) \
             if os.path.isdir('static/graphs') else []
    return render_template('admin/dashboard.html',user=user,complaints=complaints,
                           stats=stats,cat_data=cat_data,audit_logs=audit_logs,graphs=graphs)

@app.route('/admin/analytics')
@login_required
@role_required('admin')
def analytics():
    graphs = sorted(f for f in os.listdir('static/graphs') if f.endswith('.png')) \
             if os.path.isdir('static/graphs') else []
    graph_titles = {
        'eda_analysis.png':('Exploratory Data Analysis','bi-bar-chart-line'),
        'heatmap.png':('SLA Violation Heatmap','bi-grid-3x3'),
        'trends.png':('Monthly Trends','bi-calendar-range'),
        'learning_curves.png':('Learning Curves','bi-activity'),
        'confusion_matrix.png':('Confusion Matrix','bi-grid'),
        'roc_curves.png':('ROC Curves','bi-graph-up'),
        'model_comparison.png':('Model Comparison','bi-trophy'),
        'feature_importance.png':('Feature Importance','bi-bar-chart'),
        'cv_scores.png':('Cross-Validation Scores','bi-shield-check'),
        'regression_analysis.png':('Regression Analysis','bi-bullseye'),
    }
    return render_template('admin/analytics.html',graphs=graphs,graph_titles=graph_titles)

# ── API ───────────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    cat,pri,dist,desc = (data.get(k,'') for k in ('category','priority','district','description'))
    if not cat: cat = 'Sanitation'
    if not pri: pri = 'Medium'
    if not dist: dist = 'Ernakulam'
    dept = DEPARTMENTS.get(cat,'Municipality')
    risk,label,days = predict_sla(cat,dept,pri,dist,desc,data.get('resubmission',False))
    return jsonify({'risk_score':risk,'risk_label':label,'predicted_days':days,
                    'sla_threshold':SLA_THRESHOLDS.get(pri,10)})

# ══════════════════════════════════════════════════════════════════════════════
# SEED
# ══════════════════════════════════════════════════════════════════════════════

def seed_db():
    with app.test_request_context():
        if query_db('SELECT id FROM users LIMIT 1',one=True): return
        admin = modify_db('INSERT INTO users (name,email,password,role,district) VALUES (?,?,?,?,?)',
                          ['Admin Kerala','admin@kerala.gov.in',generate_password_hash('admin123'),'admin','Ernakulam'])
        depts = list(set(DEPARTMENTS.values()))
        off_map = {}
        for i,dept in enumerate(depts):
            oid = modify_db('INSERT INTO users (name,email,password,role,department,district,experience_years) VALUES (?,?,?,?,?,?,?)',
                            [f'Officer {i+1}',f'officer{i+1}@kerala.gov.in',generate_password_hash('officer123'),
                             'official',dept,random.choice(DISTRICTS),random.randint(3,18)])
            off_map[dept] = oid
        cit_ids = []
        for i in range(5):
            cid = modify_db('INSERT INTO users (name,email,password,role,district) VALUES (?,?,?,?,?)',
                            [f'Citizen {i+1}',f'citizen{i+1}@example.com',generate_password_hash('citizen123'),
                             'citizen',random.choice(DISTRICTS)])
            cit_ids.append(cid)
        samples = [
            ('Housing','High','Ernakulam','Water leakage in government housing unit causing severe property damage'),
            ('Sanitation','Critical','Kozhikode','Open sewage near school premises — urgent health hazard'),
            ('Healthcare','Medium','Thrissur','Essential medicines unavailable at PHC for two weeks'),
            ('Revenue','Low','Kannur','Land registration certificate pending 30 days without update'),
            ('Public Works','High','Malappuram','Main road severely damaged after monsoon rains causing accidents'),
            ('Education','Medium','Palakkad','Mid-day meal irregularities in three government schools'),
            ('Social Security','High','Wayanad','Pension not credited to elderly widow for 3 months'),
            ('Water Supply','Critical','Alappuzha','No water supply in entire ward for 5 days'),
        ]
        now = datetime.utcnow()
        statuses = ['Pending','In Progress','Resolved','Escalated']
        for i,(cat,pri,dist,desc) in enumerate(samples):
            dept = DEPARTMENTS[cat]; sla = SLA_THRESHOLDS[pri]
            risk,label,days = predict_sla(cat,dept,pri,dist,desc)
            days_ago = random.randint(1,20)
            sub = (now-timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
            stat = statuses[i%len(statuses)]
            esc = 1 if stat=='Escalated' else 0
            res = (now-timedelta(days=max(1,days_ago-8))).strftime('%Y-%m-%d %H:%M:%S') if stat=='Resolved' else None
            esc_at = (now-timedelta(days=max(0,days_ago-sla))).strftime('%Y-%m-%d %H:%M:%S') if esc else None
            nid = modify_db(
                '''INSERT INTO complaints (complaint_id,user_id,assigned_to,category,department,priority,
                   district,description,sla_threshold,predicted_days,sla_risk_score,sla_risk_label,
                   status,escalated,escalated_at,resolved_at,submitted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                [f'KL2024-{str(i+1).zfill(5)}',cit_ids[i%5],off_map.get(dept),
                 cat,dept,pri,dist,desc,sla,days,risk,label,stat,esc,esc_at,res,sub])
            log_action(nid,'Complaint submitted (demo)',cit_ids[i%5])
    print("✔ DB seeded")

# ══════════════════════════════════════════════════════════════════════════════
# JINJA FILTERS
# ══════════════════════════════════════════════════════════════════════════════

@app.template_filter('tojson')
def to_json_filter(x):
    return json.dumps(x)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    seed_db()
    load_models()
    print("\n  ➜ Open http://127.0.0.1:5000")
    print("  Admin: admin@kerala.gov.in / admin123")
    print("  Official: officer1@kerala.gov.in / officer123")
    print("  Citizen: citizen1@example.com / citizen123\n")
    app.run(debug=True, port=5000)
