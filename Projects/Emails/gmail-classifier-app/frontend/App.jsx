import React, { useState, useEffect } from 'react';
import { Mail, Briefcase, BarChart3, RefreshCw, Eye, CheckCircle, XCircle, MessageSquare, ArrowRightLeft, EyeOff, Download } from 'lucide-react';

const API_URL = 'http://localhost:8001/api';

const decodeHtml = (text) => {
  if (!text) return '';
  const el = document.createElement('textarea');
  el.innerHTML = text;
  return el.value;
};

// Job application tracking categories
const CATEGORIES = [
  { key: 'job_applied', label: 'Applied', icon: CheckCircle, color: 'blue', description: 'Applications sent' },
  { key: 'job_rejected', label: 'Rejected', icon: XCircle, color: 'red', description: 'Rejections received' },
  { key: 'job_followup', label: 'Followups', icon: MessageSquare, color: 'green', description: 'Action needed' },
];

const colorClasses = {
  blue: 'bg-blue-50 text-blue-600 border-blue-200',
  red: 'bg-red-50 text-red-600 border-red-200',
  green: 'bg-green-50 text-green-600 border-green-200',
};

function App() {
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(null);
  const abortControllerRef = React.useRef(null);
  const [categories, setCategories] = useState({});
  const [emails, setEmails] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [selectedEmail, setSelectedEmail] = useState(null);
  const [stats, setStats] = useState(null);
  const [analysisSummary, setAnalysisSummary] = useState(null);
  const [view, setView] = useState('overview'); // overview, category, email
  const [dateFrom, setDateFrom] = useState(`${new Date().getFullYear()}-01-01`);
  const [dateTo, setDateTo] = useState((() => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; })());
  const [hasSavedData, setHasSavedData] = useState(false);
  const [matches, setMatches] = useState({});

  useEffect(() => {
    setDateTo((() => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; })());
    fetchStats();
    loadSavedData();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const loadSavedData = async () => {
    try {
      const response = await fetch(`${API_URL}/saved`);
      const data = await response.json();
      if (data.total > 0) {
        setCategories(data.categories);
        setEmails(data.emails);
        setMatches(data.matches || {});
        setAnalysisSummary({
          total: data.total,
          ignored: data.ignored || 0,
          jobRelated: data.total - (data.ignored || 0),
          dateFrom,
          dateTo,
        });
        setHasSavedData(true);
        setView('overview');
      }
    } catch (error) {
      console.error('Error loading saved data:', error);
    }
  };

  const applyResult = (result) => {
    setCategories(result.categories);
    setEmails(result.emails);
    setMatches(result.matches || {});
    setAnalysisSummary({
      total: result.total,
      ignored: result.ignored || 0,
      jobRelated: result.total - (result.ignored || 0),
      dateFrom,
      dateTo,
    });
    setHasSavedData(true);
    setView('overview');
  };

  const analyzeEmails = async () => {
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setAnalyzing(true);
    const today = (() => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; })();
    setDateTo(today);
    setProgress({ phase: 'listing', message: 'Connecting to Gmail...' });
    try {
      const after = dateFrom.replaceAll('-', '/');
      const beforeDate = new Date(today);
      beforeDate.setDate(beforeDate.getDate() + 1);
      const before = beforeDate.toISOString().split('T')[0].replaceAll('-', '/');
      const response = await fetch(`${API_URL}/analyze?after=${after}&before=${before}`, {
        signal: controller.signal,
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              if (event.phase === 'done') {
                applyResult(event.result);
              } else if (event.phase === 'error') {
                alert('Error: ' + event.message);
              } else {
                setProgress(event);
              }
            } catch (e) { /* skip malformed lines */ }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Analysis aborted by user');
      } else {
        console.error('Error analyzing emails:', error);
        alert('Error analyzing emails. Make sure the backend is running.');
      }
    }
    abortControllerRef.current = null;
    setAnalyzing(false);
    setProgress(null);
  };

  const abortAnalysis = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const reclassifyEmail = async (emailId, newCategory) => {
    try {
      const response = await fetch(`${API_URL}/reclassify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_id: emailId, category: newCategory }),
      });
      const data = await response.json();
      if (response.ok) {
        setCategories(data.categories);
        setEmails(data.emails);
        setMatches(data.matches || {});
        setAnalysisSummary(prev => prev ? {
          ...prev,
          ignored: data.ignored || 0,
          jobRelated: data.total - (data.ignored || 0),
        } : null);
        // If reclassifying from EmailView, go back to category listing
        if (view === 'email') {
          setSelectedEmail(null);
          setView('category');
        }
      }
    } catch (error) {
      console.error('Error reclassifying email:', error);
    }
  };

  const viewEmail = (emailId) => {
    const localEmail = emails.find(e => e.id === emailId);
    if (localEmail) {
      setSelectedEmail({
        id: localEmail.id,
        subject: localEmail.subject,
        from: localEmail.sender,
        to: '',
        date: localEmail.date,
        body: localEmail.body || localEmail.snippet,
        category: localEmail.category,
      });
      setView('email');
    }
  };

  const MoveButtons = ({ emailId, currentCategory }) => {
    const targets = [...CATEGORIES.filter(c => c.key !== currentCategory), { key: 'ignore', label: 'Ignore', icon: EyeOff }];
    return (
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-xs text-gray-400 mr-1">Move to:</span>
        {targets.map(t => {
          const Icon = t.icon;
          return (
            <button
              key={t.key}
              onClick={(e) => { e.stopPropagation(); reclassifyEmail(emailId, t.key); }}
              className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border border-gray-200 bg-gray-50 hover:bg-gray-100 text-gray-600 hover:text-gray-900 transition"
              title={`Move to ${t.label}`}
            >
              <Icon className="w-3 h-3" />
              {t.label}
            </button>
          );
        })}
      </div>
    );
  };

  const OverviewView = () => (
    <div className="space-y-6">
      {/* Date Range + Stats */}
      {stats && analysisSummary && (
        <div className="space-y-4">
        <div className="bg-white rounded-lg shadow px-6 py-3 border-2 border-indigo-100 flex items-center gap-2 text-sm text-gray-700">
          <span className="font-medium text-indigo-600">Analyzed period:</span>
          <span>{new Date(analysisSummary.dateFrom).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>
          <span className="text-gray-400">—</span>
          <span>{new Date(analysisSummary.dateTo).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white rounded-lg shadow p-6 border-2 border-blue-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Scanned</p>
                <p className="text-3xl font-bold text-gray-900">{analysisSummary.total.toLocaleString()}</p>
              </div>
              <Mail className="w-12 h-12 text-blue-500" />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 border-2 border-green-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Job-Related</p>
                <p className="text-3xl font-bold text-gray-900">{analysisSummary.jobRelated.toLocaleString()}</p>
              </div>
              <Briefcase className="w-12 h-12 text-green-500" />
            </div>
          </div>
        </div>
        </div>
      )}

      {/* Category Cards */}
      {hasSavedData ? (
        <div>
          <h2 className="text-2xl font-bold mb-4">Job Applications</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {CATEGORIES.map(cat => {
                const Icon = cat.icon;
                const data = categories[cat.key] || { count: 0, percentage: 0 };
                return (
                  <button
                    key={cat.key}
                    onClick={() => { setSelectedCategory(cat.key); setView('category'); }}
                    className={`border-2 rounded-lg p-6 hover:shadow-lg transition text-left w-full ${colorClasses[cat.color]}`}
                  >
                    <div className="flex items-center justify-between mb-4">
                      <Icon className="w-8 h-8" />
                      <span className="text-3xl font-bold">{data.count}</span>
                    </div>
                    <div className="text-sm font-medium mb-1">{cat.label}</div>
                    <div className="text-xs opacity-75">{cat.description}</div>
                  </button>
                );
              })}
          </div>
        </div>
      ) : (
        <div className="text-center py-16 bg-white rounded-lg shadow">
          <Briefcase className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <p className="text-gray-600 mb-4">Select a date range and scan your emails for job applications</p>
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="flex flex-col items-start">
              <label className="text-xs text-gray-500 mb-1">From</label>
              <input
                type="date"
                value={dateFrom}
                min="2026-01-01"
                onChange={e => setDateFrom(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <span className="text-gray-400 mt-5">to</span>
            <div className="flex flex-col items-start">
              <label className="text-xs text-gray-500 mb-1">To</label>
              <input
                type="date"
                value={dateTo}
                onChange={e => setDateTo(e.target.value)}
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <button
            onClick={analyzeEmails}
            disabled={analyzing}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {analyzing ? 'Analyzing...' : 'Scan My Emails'}
          </button>
        </div>
      )}
    </div>
  );

  const downloadCategoryPdf = (categoryLabel, categoryEmails) => {
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const snippet = (e) => {
      const text = decodeHtml(e.snippet || (e.body || '').substring(0, 200));
      return esc(text.length > 200 ? text.substring(0, 200) + '...' : text);
    };
    const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>${categoryLabel} - Job Application Tracker</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; color: #1a1a1a; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  .subtitle { color: #666; font-size: 14px; margin-bottom: 30px; }
  .email { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; page-break-inside: avoid; }
  .email h3 { margin: 0 0 2px 0; font-size: 14px; }
  .email .sender { color: #4b5563; font-size: 12px; margin-bottom: 4px; }
  .email .snippet { color: #6b7280; font-size: 12px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .email .date { color: #9ca3af; font-size: 11px; margin-top: 6px; }
  .badge { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px; margin-left: 6px; }
  .badge-rejected { background: #fee2e2; color: #dc2626; }
  .badge-followup { background: #dcfce7; color: #16a34a; }
  .remark { font-size: 11px; margin-top: 2px; }
  .remark-rejected { color: #ef4444; }
  .remark-followup { color: #16a34a; }
  @media print { body { padding: 0; } .email { border-color: #ccc; } }
</style></head><body>
<h1>${categoryLabel}</h1>
<div class="subtitle">${categoryEmails.length} emails - Generated ${new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</div>
${categoryEmails.map(e => {
  const match = matches[e.id];
  const badge = match ? `<span class="badge badge-${match.status}">${match.status === 'rejected' ? 'Rejected' : 'Followup'}</span>` : '';
  const remark = match?.remark ? `<div class="remark remark-${match.status}">${esc(match.remark)}</div>` : '';
  return `<div class="email">
  <h3>${esc(e.subject)}${badge}</h3>
  <div class="sender">${esc(e.sender.split('<')[0].trim())}</div>
  <div class="snippet">${snippet(e)}</div>
  <div class="date">${esc(e.date)}</div>
  ${remark}
</div>`;
}).join('\n')}
</body></html>`;

    const win = window.open('', '_blank');
    win.document.write(html);
    win.document.close();
    setTimeout(() => win.print(), 500);
  };

  const CategoryView = () => {
    const categoryEmails = emails.filter(e => e.category === selectedCategory);
    const catConfig = CATEGORIES.find(c => c.key === selectedCategory);
    const Icon = catConfig?.icon || Mail;
    const isAppliedView = selectedCategory === 'job_applied';

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setView('overview')}
            className="text-gray-600 hover:text-gray-900"
          >
            ← Back
          </button>
          <Icon className="w-8 h-8 text-gray-700" />
          <h2 className="text-2xl font-bold">{catConfig?.label || selectedCategory}</h2>
          <span className="bg-gray-100 px-3 py-1 rounded-full text-sm font-medium">
            {categoryEmails.length} emails
          </span>
          <button
            onClick={() => downloadCategoryPdf(catConfig?.label || selectedCategory, categoryEmails)}
            className="ml-auto flex items-center gap-1.5 bg-gray-100 text-gray-700 px-3 py-1.5 rounded-lg hover:bg-gray-200 transition text-sm font-medium"
            title="Download as PDF"
          >
            <Download className="w-4 h-4" />
            Download PDF
          </button>
        </div>


        <div className="space-y-3">
          {categoryEmails.map(email => {
            const match = isAppliedView ? matches[email.id] : null;
            return (
              <div
                key={email.id}
                className="rounded-lg shadow p-4 hover:shadow-md transition border-2 bg-white border-transparent hover:border-blue-200"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 cursor-pointer" onClick={() => viewEmail(email.id)}>
                    <h3 className="font-semibold text-gray-900 mb-1">
                      {email.subject}
                      {match && (
                        <span
                          onClick={(e) => { e.stopPropagation(); viewEmail(match.matched_email_id); }}
                          className={`ml-2 text-xs font-normal px-1.5 py-0.5 rounded cursor-pointer hover:underline ${
                            match.status === 'rejected' ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                          }`}
                          title={`View ${match.status} email`}
                        >
                          {match.status === 'rejected' ? 'Rejected' : 'Followup'} →
                        </span>
                      )}
                    </h3>
                    <p className="text-sm text-gray-600">{email.sender.split('<')[0].trim()}</p>
                  </div>
                  <Eye className="w-5 h-5 text-gray-400 cursor-pointer shrink-0 ml-2" onClick={() => viewEmail(email.id)} />
                </div>
                <p className="text-sm text-gray-500 line-clamp-2 cursor-pointer" onClick={() => viewEmail(email.id)}>{email.snippet}</p>
                <div className="flex items-center justify-between mt-3 pt-2 border-t border-gray-100">
                  <div>
                    <p className="text-xs text-gray-400">{email.date}</p>
                    {match?.remark && (
                      <p className={`text-xs mt-0.5 ${match.status === 'rejected' ? 'text-red-500' : 'text-green-600'}`}>
                        {match.remark}
                      </p>
                    )}
                  </div>
                  <MoveButtons emailId={email.id} currentCategory={email.category} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const EmailView = () => {
    if (!selectedEmail) return null;

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              setSelectedEmail(null);
              setView('category');
            }}
            className="text-gray-600 hover:text-gray-900"
          >
            ← Back
          </button>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="flex items-start justify-between mb-4">
            <h1 className="text-2xl font-bold flex-1">{selectedEmail.subject}</h1>
          </div>

          {selectedEmail.category && (
            <div className="mb-4 pb-4 border-b">
              <MoveButtons emailId={selectedEmail.id} currentCategory={selectedEmail.category} />
            </div>
          )}

          <div className="space-y-2 mb-6 pb-6 border-b">
            <div className="flex items-start">
              <span className="text-sm text-gray-600 w-20">From:</span>
              <span className="text-sm text-gray-900">{selectedEmail.from}</span>
            </div>
            <div className="flex items-start">
              <span className="text-sm text-gray-600 w-20">To:</span>
              <span className="text-sm text-gray-900">{selectedEmail.to}</span>
            </div>
            <div className="flex items-start">
              <span className="text-sm text-gray-600 w-20">Date:</span>
              <span className="text-sm text-gray-900">{selectedEmail.date}</span>
            </div>
          </div>

          <div className="max-w-none overflow-hidden">
            <div className="whitespace-pre-wrap break-words overflow-wrap-anywhere font-sans text-sm text-gray-700 leading-relaxed" style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}>
              {decodeHtml(selectedEmail.body)}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Briefcase className="w-10 h-10 text-blue-600" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Job Application Tracker</h1>
                <p className="text-sm text-gray-600">Track your job applications from email</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {stats && (
                <span className="text-sm text-gray-600">
                  📧 {stats.email_address}
                </span>
              )}
              {emails.length > 0 && view === 'overview' && (
                <div className="flex items-center gap-3">
                  <input
                    type="date"
                    value={dateFrom}
                    min="2026-01-01"
                    onChange={e => setDateFrom(e.target.value)}
                    className="border border-gray-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-gray-400 text-sm">to</span>
                  <input
                    type="date"
                    value={dateTo}
                    onChange={e => setDateTo(e.target.value)}
                    className="border border-gray-300 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={analyzeEmails}
                    disabled={analyzing}
                    className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    <RefreshCw className={`w-4 h-4 ${analyzing ? 'animate-spin' : ''}`} />
                    {hasSavedData ? 'Check for New' : 'Analyze'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Progress Indicator */}
      {analyzing && progress && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
          <div className="bg-white rounded-lg shadow p-6 border-2 border-blue-100">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <RefreshCw className="w-5 h-5 text-blue-600 animate-spin" />
                <span className="font-medium text-gray-900">{progress.message}</span>
              </div>
              <button
                onClick={abortAnalysis}
                className="flex items-center gap-1.5 bg-red-50 text-red-600 border border-red-200 px-3 py-1.5 rounded-lg hover:bg-red-100 transition text-sm font-medium"
              >
                Abort
              </button>
            </div>
            {(progress.phase === 'analyzing' || progress.phase === 'classifying') && progress.total > 0 && (
              <>
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${(progress.processed / progress.total) * 100}%` }}
                  />
                </div>
                <div className="flex items-center justify-between mt-2">
                  <p className="text-sm text-gray-500">
                    {progress.processed} / {progress.total} emails ({Math.round((progress.processed / progress.total) * 100)}%)
                  </p>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-green-600">{progress.important || 0} job-related</span>
                    <span className="text-slate-400">{progress.ignored || 0} ignored</span>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {view === 'overview' && <OverviewView />}
        {view === 'category' && <CategoryView />}
        {view === 'email' && <EmailView />}
      </main>

      {/* Footer */}
      <footer className="mt-16 pb-8 text-center text-sm text-gray-500">
        <p>Job Application Tracker - Track your applications from email</p>
      </footer>
    </div>
  );
}

export default App;
