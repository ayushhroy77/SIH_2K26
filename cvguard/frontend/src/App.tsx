import React, { useState, useEffect } from 'react';
import {
  Shield,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  FileText,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  RefreshCw,
  Upload,
  Info,
  Layers,
  Activity,
  Zap,
  Target,
  Database,
  Hash,
} from 'lucide-react';

interface Finding {
  finding_id: string;
  asset_type: string;
  asset_ref: string;
  detector: string;
  reason: string;
  evidence: string[];
  confidence: number;
  severity: string;
  disposition: string;
  assumptions: string[];
  limitations: string[];
  created_at: string;
}

interface SignedFinding {
  ledger_id: number;
  entry_hash: string;
  prev_hash: string;
  signature: string;
  finding: Finding;
  created_at: string;
}

interface AuditVerifyResult {
  valid: boolean;
  entries_checked: number;
  first_invalid_entry_id: number | null;
  reason?: string | null;
}

// Configurable gateway base URL
const GATEWAY_URL =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_GATEWAY_URL ||
  'http://localhost:8000';

export default function App() {
  const [findings, setFindings] = useState<SignedFinding[]>([]);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [assetTypeFilter, setAssetTypeFilter] = useState<string>('all');
  const [detectorFilter, setDetectorFilter] = useState<string>('all');

  // Audit state
  const [auditResult, setAuditResult] = useState<AuditVerifyResult | null>(null);
  const [verifyingAudit, setVerifyingAudit] = useState<boolean>(false);

  // Ingest upload state
  const [showUpload, setShowUpload] = useState<boolean>(false);
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null);
  const [contributorId, setContributorId] = useState<string>('contributor-alpha');
  const [datasetId, setDatasetId] = useState<string>('default');
  const [labels, setLabels] = useState<string>('');
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  // Reference distributions
  const [showRefDist, setShowRefDist] = useState<boolean>(false);
  const [refDists, setRefDists] = useState<any[]>([]);

  const fetchFindings = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (severityFilter !== 'all') params.append('severity', severityFilter);
      if (assetTypeFilter !== 'all') params.append('asset_type', assetTypeFilter);

      const res = await fetch(`${GATEWAY_URL}/findings?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`Failed to fetch findings: HTTP ${res.status}`);
      }
      const data: SignedFinding[] = await res.json();

      if (detectorFilter !== 'all') {
        setFindings(data.filter((f) => f.finding.detector.includes(detectorFilter)));
      } else {
        setFindings(data);
      }
    } catch (err: any) {
      setError(err.message || 'Unable to connect to Gateway service.');
    } finally {
      setLoading(false);
    }
  };

  const fetchRefDistributions = async () => {
    try {
      const res = await fetch(`${GATEWAY_URL}/reference-distributions`);
      if (res.ok) {
        const data = await res.json();
        setRefDists(data.reference_distributions || []);
      }
    } catch {
      // Ignore if offline
    }
  };

  const verifyAuditLedger = async () => {
    setVerifyingAudit(true);
    try {
      const res = await fetch(`${GATEWAY_URL}/audit/verify`);
      if (!res.ok) throw new Error(`Audit verification failed with HTTP ${res.status}`);
      const data = await res.json();
      setAuditResult(data);
    } catch (err: any) {
      setAuditResult({
        valid: false,
        entries_checked: 0,
        first_invalid_entry_id: null,
        reason: err.message,
      });
    } finally {
      setVerifyingAudit(false);
    }
  };

  useEffect(() => {
    fetchFindings();
    verifyAuditLedger();
    fetchRefDistributions();
  }, [severityFilter, assetTypeFilter, detectorFilter]);

  const toggleRow = (ledgerId: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(ledgerId)) next.delete(ledgerId);
      else next.add(ledgerId);
      return next;
    });
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFiles || uploadFiles.length === 0) {
      setUploadMessage('Please select at least one image file.');
      return;
    }

    setUploading(true);
    setUploadMessage(null);

    const formData = new FormData();
    for (let i = 0; i < uploadFiles.length; i++) {
      formData.append('files', uploadFiles[i]);
    }
    formData.append('contributor_id', contributorId);
    formData.append('dataset_id', datasetId);
    if (labels) formData.append('labels', labels);

    try {
      const res = await fetch(`${GATEWAY_URL}/ingest/images`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Ingestion failed (HTTP ${res.status}): ${errText}`);
      }

      const result = await res.json();
      setUploadMessage(
        `Ingested ${result.ingested_count} image(s). Generated ${result.findings_count} finding(s) sealed into governance.`
      );
      setUploadFiles(null);
      await fetchFindings();
      await verifyAuditLedger();
    } catch (err: any) {
      setUploadMessage(`Error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const renderDetectorBadge = (detector: string) => {
    if (detector.includes('phash')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-mono bg-sky-950/80 text-sky-300 border border-sky-800 rounded">
          <Hash className="w-3 h-3 text-sky-400" />
          pHash Duplicate
        </span>
      );
    }
    if (detector.includes('ood')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-mono bg-purple-950/80 text-purple-300 border border-purple-800 rounded">
          <Activity className="w-3 h-3 text-purple-400" />
          OOD Mahalanobis
        </span>
      );
    }
    if (detector.includes('label_flip')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-mono bg-amber-950/80 text-amber-300 border border-amber-800 rounded">
          <Target className="w-3 h-3 text-amber-400" />
          Label-Flip kNN
        </span>
      );
    }
    if (detector.includes('trigger')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-mono bg-rose-950/80 text-rose-300 border border-rose-800 rounded">
          <Zap className="w-3 h-3 text-rose-400" />
          FFT Trigger Backdoor
        </span>
      );
    }
    if (detector.includes('source_aggregator')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-mono bg-indigo-950/80 text-indigo-300 border border-indigo-800 rounded">
          <Layers className="w-3 h-3 text-indigo-400" />
          Source Aggregator
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-mono bg-neutral-800 text-neutral-300 border border-neutral-700 rounded">
        {detector.split('.').pop() || detector}
      </span>
    );
  };

  const renderSeverityBadge = (severity: string) => {
    const s = severity.toUpperCase();
    switch (s) {
      case 'CRITICAL':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-bold bg-rose-950 text-rose-300 border border-rose-800 rounded">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
            [CRIT] CRITICAL
          </span>
        );
      case 'HIGH':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-semibold bg-amber-950 text-amber-300 border border-amber-800 rounded">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            [HIGH] HIGH
          </span>
        );
      case 'MEDIUM':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-medium bg-yellow-950 text-yellow-300 border border-yellow-800 rounded">
            <Info className="w-3.5 h-3.5 text-yellow-400" />
            [MED] MEDIUM
          </span>
        );
      case 'LOW':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-medium bg-emerald-950 text-emerald-300 border border-emerald-800 rounded">
            <Info className="w-3.5 h-3.5 text-emerald-400" />
            [LOW] LOW
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-medium bg-neutral-800 text-neutral-300 border border-neutral-700 rounded">
            <Info className="w-3.5 h-3.5 text-neutral-400" />
            [INFO] INFO
          </span>
        );
    }
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6 font-sans">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header Bar */}
        <header className="border border-neutral-800 bg-neutral-900/60 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-emerald-950/80 border border-emerald-800/80 rounded-lg">
              <Shield className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-neutral-100">CVGuard</h1>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                  PHASE 3: COMPLETE DATA PLANE
                </span>
              </div>
              <p className="text-xs text-neutral-400">
                pHash Near-Duplicate &bull; OOD Mahalanobis &bull; kNN Label-Flip &bull; 2D FFT Spectral Trigger &bull; Unified Source Aggregator
              </p>
            </div>
          </div>

          {/* Audit Status & Action Controls */}
          <div className="flex items-center flex-wrap gap-3">
            {auditResult && (
              <div
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs font-mono ${
                  auditResult.valid
                    ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                    : 'bg-rose-950/60 border-rose-800 text-rose-300'
                }`}
              >
                {auditResult.valid ? (
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                ) : (
                  <ShieldAlert className="w-4 h-4 text-rose-400" />
                )}
                <span>
                  CHAIN: {auditResult.valid ? 'VALID' : 'TAMPERED'} ({auditResult.entries_checked} entries)
                </span>
              </div>
            )}

            <button
              onClick={() => setShowRefDist(!showRefDist)}
              className="flex items-center space-x-1.5 px-3 py-1.5 text-xs bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 rounded-lg text-neutral-200 transition"
              title="View class reference distributions for OOD"
            >
              <Database className="w-3.5 h-3.5 text-purple-400" />
              <span>Reference Distributions</span>
            </button>

            <button
              onClick={verifyAuditLedger}
              disabled={verifyingAudit}
              className="flex items-center space-x-1.5 px-3 py-1.5 text-xs bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 rounded-lg text-neutral-200 transition"
              title="Verify cryptographic hash chain & signatures"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${verifyingAudit ? 'animate-spin' : ''}`} />
              <span>Verify Audit</span>
            </button>

            <button
              onClick={() => setShowUpload(!showUpload)}
              className="flex items-center space-x-1.5 px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 font-medium rounded-lg text-white transition"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Ingest Batch</span>
            </button>
          </div>
        </header>

        {/* Reference Distributions Drawer */}
        {showRefDist && (
          <section className="border border-purple-900/50 bg-purple-950/20 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-purple-900/40 pb-2">
              <div className="flex items-center space-x-2">
                <Database className="w-4 h-4 text-purple-400" />
                <h2 className="text-sm font-semibold text-purple-200">
                  Registered Class Reference Distributions (OOD Centroids & Precision Matrices)
                </h2>
              </div>
              <button
                onClick={() => setShowRefDist(false)}
                className="text-xs text-neutral-400 hover:text-neutral-200"
              >
                Close
              </button>
            </div>
            {refDists.length === 0 ? (
              <p className="text-xs text-neutral-400 italic">
                No class reference distributions registered yet. Submit via POST /reference-distributions.
              </p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {refDists.map((d, i) => (
                  <div key={i} className="p-3 bg-neutral-900/80 border border-neutral-800 rounded-lg text-xs space-y-1">
                    <div className="font-bold text-neutral-200 flex justify-between">
                      <span>Class: {d.class_name}</span>
                      <span className="text-[10px] text-purple-400 font-mono">Dataset: {d.dataset_id}</span>
                    </div>
                    <div className="text-[11px] text-neutral-400">Reference Samples: {d.num_samples}</div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Ingest Drawer / Section */}
        {showUpload && (
          <section className="border border-neutral-800 bg-neutral-900/90 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
              <div className="flex items-center space-x-2">
                <Upload className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-neutral-200">
                  Data Plane Vision Batch Ingest
                </h2>
              </div>
              <button
                onClick={() => setShowUpload(false)}
                className="text-xs text-neutral-400 hover:text-neutral-200"
              >
                Close
              </button>
            </div>

            <form onSubmit={handleUploadSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs font-medium text-neutral-300 mb-1">
                    Select Images (JPEG / PNG)
                  </label>
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={(e) => setUploadFiles(e.target.files)}
                    className="w-full text-xs text-neutral-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-neutral-800 file:text-neutral-200 hover:file:bg-neutral-700 cursor-pointer"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-neutral-300 mb-1">
                    Default Contributor ID
                  </label>
                  <input
                    type="text"
                    value={contributorId}
                    onChange={(e) => setContributorId(e.target.value)}
                    placeholder="e.g. contributor-alpha"
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-3 py-1.5 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-neutral-300 mb-1">
                    Dataset & Class Labels (Optional, comma-separated)
                  </label>
                  <input
                    type="text"
                    value={labels}
                    onChange={(e) => setLabels(e.target.value)}
                    placeholder="e.g. vehicle, vehicle, cat"
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-3 py-1.5 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-[11px] text-neutral-400">
                  Ingests images to MinIO, calculates pHash + embeddings, runs 4 detectors, aggregates sources, and seals findings into Governance.
                </span>
                <button
                  type="submit"
                  disabled={uploading}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition"
                >
                  {uploading && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                  <span>{uploading ? 'Processing Batch...' : 'Submit Batch'}</span>
                </button>
              </div>

              {uploadMessage && (
                <div className="p-3 bg-neutral-950 border border-neutral-800 rounded-lg text-xs font-mono text-neutral-300">
                  {uploadMessage}
                </div>
              )}
            </form>
          </section>
        )}

        {/* Filter Controls Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 bg-neutral-900/40 border border-neutral-800/80 rounded-xl px-4 py-3">
          <div className="flex items-center flex-wrap gap-3 text-xs">
            <span className="text-neutral-400 font-medium">Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="bg-neutral-900 border border-neutral-700 rounded px-2.5 py-1 text-neutral-200 text-xs"
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="info">Info</option>
            </select>

            <span className="text-neutral-400 font-medium ml-2">Detector:</span>
            <select
              value={detectorFilter}
              onChange={(e) => setDetectorFilter(e.target.value)}
              className="bg-neutral-900 border border-neutral-700 rounded px-2.5 py-1 text-neutral-200 text-xs"
            >
              <option value="all">All Detectors (5 Active)</option>
              <option value="phash">pHash Near-Duplicate</option>
              <option value="ood">OOD Mahalanobis</option>
              <option value="label_flip">kNN Label-Flip</option>
              <option value="trigger">FFT Spectral Trigger</option>
              <option value="source_aggregator">Unified Source Aggregator</option>
            </select>

            <span className="text-neutral-400 font-medium ml-2">Asset Type:</span>
            <select
              value={assetTypeFilter}
              onChange={(e) => setAssetTypeFilter(e.target.value)}
              className="bg-neutral-900 border border-neutral-700 rounded px-2.5 py-1 text-neutral-200 text-xs"
            >
              <option value="all">All Types</option>
              <option value="sample">Sample</option>
              <option value="source">Source</option>
            </select>
          </div>

          <div className="flex items-center space-x-2 text-xs text-neutral-400">
            <span>Total Findings: {findings.length}</span>
            <button
              onClick={fetchFindings}
              className="p-1.5 hover:bg-neutral-800 rounded transition"
              title="Refresh findings list"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 bg-rose-950/80 border border-rose-800 rounded-xl text-rose-300 text-xs font-mono flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-rose-400" />
              <span>{error}</span>
            </div>
            <button
              onClick={fetchFindings}
              className="px-2.5 py-1 bg-rose-900 hover:bg-rose-800 rounded text-rose-100 text-xs"
            >
              Retry
            </button>
          </div>
        )}

        {/* Findings Table */}
        <div className="border border-neutral-800 rounded-xl overflow-hidden bg-neutral-900/30">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-neutral-900/90 border-b border-neutral-800 text-neutral-400 uppercase font-mono tracking-wider">
                  <th className="py-3 px-4 w-12 text-center">#</th>
                  <th className="py-3 px-4 w-32">Severity</th>
                  <th className="py-3 px-4 w-44">Detector</th>
                  <th className="py-3 px-4">Reason & Target</th>
                  <th className="py-3 px-4 w-28 text-center">Disposition</th>
                  <th className="py-3 px-4 w-24 text-right">Confidence</th>
                  <th className="py-3 px-4 w-36 text-right">Created At</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60 font-mono">
                {loading ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-neutral-400">
                      <div className="flex items-center justify-center space-x-2">
                        <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                        <span>Loading findings from Governance Spine...</span>
                      </div>
                    </td>
                  </tr>
                ) : findings.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-neutral-500">
                      No findings match the current filter criteria.
                    </td>
                  </tr>
                ) : (
                  findings.map((item) => {
                    const isExpanded = expandedIds.has(item.ledger_id);
                    return (
                      <React.Fragment key={item.ledger_id}>
                        <tr
                          onClick={() => toggleRow(item.ledger_id)}
                          className="hover:bg-neutral-800/40 cursor-pointer transition border-l-2 border-l-transparent hover:border-l-emerald-500"
                        >
                          <td className="py-3 px-4 text-center text-neutral-500">
                            {isExpanded ? (
                              <ChevronDown className="w-3.5 h-3.5 inline" />
                            ) : (
                              <ChevronRight className="w-3.5 h-3.5 inline" />
                            )}
                            <span className="ml-1">{item.ledger_id}</span>
                          </td>
                          <td className="py-3 px-4">{renderSeverityBadge(item.finding.severity)}</td>
                          <td className="py-3 px-4">{renderDetectorBadge(item.finding.detector)}</td>
                          <td className="py-3 px-4 text-neutral-200 font-sans">
                            <div className="font-semibold text-xs line-clamp-1">{item.finding.reason}</div>
                            <div className="text-[11px] font-mono text-neutral-400">
                              {item.finding.asset_type}: {item.finding.asset_ref}
                            </div>
                          </td>
                          <td className="py-3 px-4 text-center">
                            <span
                              className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                                item.finding.disposition === 'QUARANTINE'
                                  ? 'bg-rose-950/80 text-rose-300 border border-rose-800'
                                  : item.finding.disposition === 'REJECT'
                                  ? 'bg-orange-950/80 text-orange-300 border border-orange-800'
                                  : 'bg-yellow-950/80 text-yellow-300 border border-yellow-800'
                              }`}
                            >
                              {item.finding.disposition}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-right text-emerald-400 font-bold">
                            {(item.finding.confidence * 100).toFixed(1)}%
                          </td>
                          <td className="py-3 px-4 text-right text-neutral-400 text-[11px]">
                            {new Date(item.created_at).toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit',
                              second: '2-digit',
                            })}
                          </td>
                        </tr>

                        {/* Expanded Detail View */}
                        {isExpanded && (
                          <tr className="bg-neutral-950/80">
                            <td colSpan={7} className="p-5 border-t border-neutral-800">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 font-sans">
                                {/* Left Column: Cryptographic Ledger Envelope */}
                                <div className="space-y-4">
                                  <div className="border border-neutral-800 rounded-lg p-3.5 bg-neutral-900/60 space-y-2">
                                    <h4 className="text-xs font-semibold text-neutral-300 flex items-center space-x-1.5">
                                      <Shield className="w-3.5 h-3.5 text-emerald-400" />
                                      <span>Cryptographic Ledger Envelope</span>
                                    </h4>
                                    <div className="text-[11px] font-mono space-y-1.5 text-neutral-400">
                                      <div>
                                        <span className="text-neutral-500">Finding ID:</span>{' '}
                                        <span className="text-neutral-200">{item.finding.finding_id}</span>
                                      </div>
                                      <div className="truncate">
                                        <span className="text-neutral-500">Entry Hash:</span>{' '}
                                        <span className="text-neutral-300">{item.entry_hash}</span>
                                      </div>
                                      <div className="truncate">
                                        <span className="text-neutral-500">Prev Hash:</span>{' '}
                                        <span className="text-neutral-300">{item.prev_hash}</span>
                                      </div>
                                      <div className="truncate">
                                        <span className="text-neutral-500">Ed25519 Sig:</span>{' '}
                                        <span className="text-emerald-400/90">{item.signature}</span>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="space-y-2">
                                    <h4 className="text-xs font-semibold text-neutral-300">Methodological Disclaimers</h4>
                                    <div className="text-[11px] text-neutral-400 space-y-1 bg-neutral-900/30 p-2.5 rounded border border-neutral-800/80">
                                      <div>
                                        <strong className="text-neutral-300">Assumptions:</strong>{' '}
                                        {item.finding.assumptions?.join('; ') || 'None specified.'}
                                      </div>
                                      <div>
                                        <strong className="text-neutral-300">Limitations:</strong>{' '}
                                        {item.finding.limitations?.join('; ') || 'None specified.'}
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                {/* Right Column: Evidence Inspection & Proxied Imagery */}
                                <div className="space-y-3">
                                  <h4 className="text-xs font-semibold text-neutral-300 flex items-center space-x-1.5">
                                    <FileText className="w-3.5 h-3.5 text-sky-400" />
                                    <span>Evidence Breakdown & Artifact Inspection</span>
                                  </h4>

                                  {/* Evidence Keys / Metadata */}
                                  <div className="bg-neutral-900/50 p-2.5 rounded-lg border border-neutral-800 text-[11px] font-mono space-y-1 text-neutral-300">
                                    {item.finding.evidence?.map((ev, evIdx) => (
                                      <div key={evIdx} className="break-all text-neutral-400">
                                        &bull; {ev}
                                      </div>
                                    ))}
                                  </div>

                                  {/* Proxied Evidence Image Viewers */}
                                  <div className="flex flex-wrap gap-3 pt-2">
                                    {item.finding.evidence
                                      ?.filter(
                                        (e) =>
                                          (e.endsWith('.jpg') ||
                                            e.endsWith('.jpeg') ||
                                            e.endsWith('.png') ||
                                            e.startsWith('sample:')) &&
                                          !e.startsWith('declared_') &&
                                          !e.startsWith('dominant_')
                                      )
                                      .map((imageRef, imgIdx) => {
                                        const cleanKey = imageRef.replace('sample:', '').trim();
                                        const proxiedUrl = `${GATEWAY_URL}/images/${cleanKey}`;
                                        return (
                                          <div
                                            key={imgIdx}
                                            className="border border-neutral-800 rounded-lg overflow-hidden bg-neutral-900/90 text-center p-2 space-y-1.5"
                                          >
                                            <div className="w-36 h-36 bg-neutral-950 flex items-center justify-center overflow-hidden rounded">
                                              <img
                                                src={proxiedUrl}
                                                alt={`Evidence ${cleanKey}`}
                                                className="w-full h-full object-contain"
                                                referrerPolicy="no-referrer"
                                                onError={(e) => {
                                                  // Fallback for non-image evidence strings
                                                  (e.target as HTMLElement).style.display = 'none';
                                                }}
                                              />
                                            </div>
                                            <div className="text-[10px] font-mono text-neutral-400 truncate max-w-[144px]">
                                              {cleanKey}
                                            </div>
                                            <a
                                              href={proxiedUrl}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              className="inline-flex items-center space-x-1 text-[10px] text-emerald-400 hover:text-emerald-300"
                                            >
                                              <span>Inspect</span>
                                              <ExternalLink className="w-2.5 h-2.5" />
                                            </a>
                                          </div>
                                        );
                                      })}
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
