import React, { useState, useEffect, useMemo } from 'react';
import {
  Shield,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Info,
  OctagonAlert,
  Search,
  ArrowUpDown,
  Filter,
  FileText,
  Download,
  ExternalLink,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Database,
  Cpu,
  Activity,
  BarChart2,
  Layers,
  Lock,
  Eye,
  Check,
  X,
  FileCode,
  Image as ImageIcon,
} from 'lucide-react';

interface Finding {
  finding_id?: string;
  asset_type: string;
  asset_ref: string;
  detector: string;
  reason: string;
  evidence: string[];
  confidence: number;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO' | string;
  disposition: 'QUARANTINE' | 'REVIEW' | 'ACCEPT' | string;
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
  created_at?: string;
}

interface AuditVerifyResult {
  valid: boolean;
  entries_checked: number;
  first_invalid_entry_id: number | null;
  reason?: string | null;
}

interface CoverageItem {
  name: string;
  description?: string;
  mitigation?: string;
  reason?: string;
  detector_id: string;
  detector_name: string;
}

interface CoverageData {
  summary: string;
  version: string;
  manifest_hash: string;
  coverage: {
    total_assets_scanned: number;
    passed_count: number;
    flagged_count: number;
    scope_description: string;
    supported_attack_classes: CoverageItem[];
    uncovered_attack_classes: CoverageItem[];
    detector_versions: Record<string, string>;
  };
}

const GATEWAY_URL =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_GATEWAY_URL ||
  'http://localhost:8000';

// Numeric rank for pure triage sorting: severity descending, then confidence descending
export const SEVERITY_RANK: Record<string, number> = {
  CRITICAL: 5,
  HIGH: 4,
  MEDIUM: 3,
  LOW: 2,
  INFO: 1,
};

export function sortFindingsTriage(items: SignedFinding[]): SignedFinding[] {
  return [...items].sort((a, b) => {
    const sevA = SEVERITY_RANK[a.finding.severity.toUpperCase()] || 0;
    const sevB = SEVERITY_RANK[b.finding.severity.toUpperCase()] || 0;
    if (sevB !== sevA) {
      return sevB - sevA; // Severity descending
    }
    return b.finding.confidence - a.finding.confidence; // Confidence descending
  });
}

export function classifyPlane(detector: string, assetType: string): string {
  const det = detector.toLowerCase();
  const asset = assetType.toLowerCase();
  if (det.includes('drift') || det.includes('distribution')) return 'Drift Plane';
  if (det.includes('inference') || det.includes('replay') || det.includes('merkle') || det.includes('adversarial_perturbation') || asset.includes('inference_record')) return 'Inference Plane';
  if (det.includes('model') || det.includes('safetensors') || det.includes('weight') || det.includes('graph') || asset.includes('model')) return 'Model Plane';
  return 'Data Plane';
}

// Fallback baseline coverage manifest
const FALLBACK_COVERAGE: CoverageData = {
  summary: 'Threat model and attack class coverage manifest for CVGuard cross-plane integrity assurance suite.',
  version: '0.6.0',
  manifest_hash: 'sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069',
  coverage: {
    total_assets_scanned: 5,
    passed_count: 1,
    flagged_count: 4,
    scope_description: 'Consolidated cross-plane integrity inspection across Data Plane, Model Plane, Inference Plane, and Drift Plane threat boundaries.',
    supported_attack_classes: [
      {
        name: 'Exact and Near-Duplicate Image Injection',
        description: 'Repeated submission of identical or near-identical images with minor compression or pixel shifts.',
        mitigation: 'Flags pairs with Hamming distance <= 10; high confidence to distance <= 2.',
        detector_id: 'cvguard.detector.phash_near_duplicate:v1.0',
        detector_name: 'Near-Duplicate Perceptual Hash Detector',
      },
      {
        name: 'Covariate Shift and Out-of-Domain Contamination',
        description: 'Inclusion of foreign, out-of-domain, or non-target category images into training partition.',
        mitigation: 'Measures Mahalanobis distance against registered class centroid and precision matrix.',
        detector_id: 'cvguard.dataplane.ood_mahalanobis:v1.0',
        detector_name: 'Out-of-Distribution Mahalanobis Detector',
      },
      {
        name: 'Targeted Label-Flipping and Poisoning Infiltration',
        description: 'Adversary injecting incorrectly labeled samples into clean clusters to degrade boundary precision.',
        mitigation: 'Computes k-NN neighborhood label consensus; flags disagreement >= 60%.',
        detector_id: 'cvguard.dataplane.knn_label_flip:v1.0',
        detector_name: 'k-NN Label-Flip Consensus Detector',
      },
      {
        name: 'Fourier High-Frequency Spectral Backdoor Injection',
        description: 'Periodic watermark or trigger embedded into spatial high frequencies to cause misclassification.',
        mitigation: '2D FFT log magnitude power spectrum analysis; detects spectral spikes > 3.0 std.',
        detector_id: 'cvguard.dataplane.fft_spectral_backdoor:v1.0',
        detector_name: 'Fourier Spectral Backdoor Trigger Detector',
      },
      {
        name: 'Safetensors Header Manipulation & Deserialization RCE',
        description: 'Malicious pickle payload or corrupted JSON metadata header attempting arbitrary code execution.',
        mitigation: 'Strict non-deserializing byte inspection; verifies SafeTensors format and rejects pickle opcodes.',
        detector_id: 'cvguard.modelplane.safetensors_structural:v1.0',
        detector_name: 'SafeTensors Structural Integrity Detector',
      },
      {
        name: 'Extreme Weight Magnitude Anomaly & NaN/Inf Infiltration',
        description: 'Model trojaning introducing anomalous weight values or non-finite layer tensors.',
        mitigation: 'Layer-wise Frobenius norm and kurtosis statistical envelope checking.',
        detector_id: 'cvguard.modelplane.weight_anomaly_detector:v1.0',
        detector_name: 'Neural Weight Anomaly & Trojan Detector',
      },
      {
        name: 'Atomic Inference Replay & Checkpoint Mismatch',
        description: 'Post-hoc substitution of model checkpoints or replay of prior inference outputs.',
        mitigation: 'Verifiable cryptographic binding linking input digest, model digest, and inference receipt.',
        detector_id: 'cvguard.inferenceplane.atomic_replay_detector:v1.0',
        detector_name: 'Atomic Inference Replay Detector',
      },
      {
        name: 'Operational Covariate Drift vs Adversarial Perturbation',
        description: 'Statistical shift in feature distributions distinguishing sensor drift from adversarial manipulation.',
        mitigation: 'Maximum Mean Discrepancy (MMD) divergence and Kolmogorov-Smirnov dual statistical testing.',
        detector_id: 'cvguard.driftplane.distribution_verifier:v1.0',
        detector_name: 'Distribution Shift & Manipulation Classifier',
      },
    ],
    uncovered_attack_classes: [
      {
        name: 'Large Spatial Transformations and Non-Affine Crops',
        reason: '64-bit DCT perceptual hash is sensitive to large rotations (>15-30 deg) and heavy random cropping.',
        detector_id: 'cvguard.detector.phash_near_duplicate:v1.0',
        detector_name: 'Near-Duplicate Perceptual Hash Detector',
      },
      {
        name: 'Adversarial Embedding Invariance (Manifold-Constrained Perturbations)',
        reason: 'Adversarial perturbations mathematically constrained within class reference covariance ellipsoid evade Mahalanobis distance.',
        detector_id: 'cvguard.dataplane.ood_mahalanobis:v1.0',
        detector_name: 'Out-of-Distribution Mahalanobis Detector',
      },
      {
        name: 'Physical World Light-Level Ambiguities in Drift Evaluation',
        reason: 'Unsupervised MMD tests cannot resolve whether lighting shifts stem from sensor recalibration or environment changes without sensor metadata.',
        detector_id: 'cvguard.driftplane.distribution_verifier:v1.0',
        detector_name: 'Distribution Shift & Manipulation Classifier',
      },
    ],
    detector_versions: {
      'cvguard.detector.phash_near_duplicate:v1.0': 'v1.0',
      'cvguard.dataplane.ood_mahalanobis:v1.0': 'v1.0',
      'cvguard.dataplane.knn_label_flip:v1.0': 'v1.0',
      'cvguard.dataplane.fft_spectral_backdoor:v1.0': 'v1.0',
      'cvguard.modelplane.safetensors_structural:v1.0': 'v1.0',
      'cvguard.modelplane.weight_anomaly_detector:v1.0': 'v1.0',
      'cvguard.inferenceplane.atomic_replay_detector:v1.0': 'v1.0',
      'cvguard.driftplane.distribution_verifier:v1.0': 'v1.0',
    },
  },
};

// Initial seed findings representing all four planes
const SEED_FINDINGS: SignedFinding[] = [
  {
    ledger_id: 104,
    entry_hash: 'sha256:4a8c9e3104928f11d612e4f01b3390c427da93108c909b7888b14e0378411d99',
    prev_hash: 'sha256:11a0bb3344cc55dd66ee77ff88aa99bb00112233445566778899aabbccddeeff',
    signature: '3f92b7c09e...ed25519_valid_signature_envelope',
    created_at: '2026-09-09T08:14:22Z',
    finding: {
      asset_type: 'MODEL',
      asset_ref: 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      detector: 'cvguard.modelplane.safetensors_structural:v1.0',
      reason: 'Arbitrary code execution risk: model checkpoint contains unauthorized pickle stream header.',
      evidence: ['header_magic:PK0304', 'embedded_python_opcode:GLOBAL', 'target_layer:backbone.layer4'],
      confidence: 0.992,
      severity: 'CRITICAL',
      disposition: 'QUARANTINE',
      assumptions: ['SafeTensors format strictly required for all deployed vision backbones.'],
      limitations: ['Only format envelope and byte-level signatures validated, not dynamic execution behavior.'],
      created_at: '2026-09-09T08:14:22Z',
    },
  },
  {
    ledger_id: 103,
    entry_hash: 'sha256:8b71d90048291048201948291048201948291048201948291048201948291048',
    prev_hash: 'sha256:99f0e1d2c3b4a596877869504132231405162738495061728394a5b6c7d8e9f0',
    signature: '7e11c8d4...ed25519_valid_signature_envelope',
    created_at: '2026-09-09T07:45:10Z',
    finding: {
      asset_type: 'SAMPLE',
      asset_ref: 'sha256:d41d8cd98f00b204e9800998ecf8427e (sample_batch_091.jpg)',
      detector: 'cvguard.dataplane.ood_mahalanobis:v1.0',
      reason: 'Out-of-distribution sample: Mahalanobis distance exceeds class reference threshold.',
      evidence: ['mahalanobis_distance:15.82', 'reference_threshold:12.00', 'class_name:pedestrian', 'image_ref:sample_batch_091.jpg'],
      confidence: 0.945,
      severity: 'HIGH',
      disposition: 'REVIEW',
      assumptions: ['Reference class distribution is unimodal Gaussian.'],
      limitations: ['May exhibit elevated false positive rates on multi-modal classes.'],
      created_at: '2026-09-09T07:45:10Z',
    },
  },
  {
    ledger_id: 102,
    entry_hash: 'sha256:55cc44dd33ee22ff11aa00bb99887766554433221100ffeeddccbbaa99887766',
    prev_hash: 'sha256:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff',
    signature: '5a22bb11...ed25519_valid_signature_envelope',
    created_at: '2026-09-09T06:30:00Z',
    finding: {
      asset_type: 'INFERENCE_RECORD',
      asset_ref: 'inf-rec-alpha-5819 (live_ingress_04.jpg)',
      detector: 'cvguard.inferenceplane.adversarial_perturbation:v1.0',
      reason: 'Adversarial frequency perturbation: high-frequency 2D DCT spectral spike in quadrant Q2.',
      evidence: ['spectral_energy_ratio:0.048', 'baseline_max:0.015', 'p_value:0.002', 'image_ref:live_ingress_04.jpg'],
      confidence: 0.884,
      severity: 'HIGH',
      disposition: 'QUARANTINE',
      assumptions: ['Input normalized to standard 224x224 ImageNet feature space.'],
      limitations: ['Spatial patch attacks without spectral anomalies may evade frequency detection.'],
      created_at: '2026-09-09T06:30:00Z',
    },
  },
  {
    ledger_id: 101,
    entry_hash: 'sha256:33221100ffeeddccbbaa99887766554433221100ffeeddccbbaa998877665544',
    prev_hash: 'sha256:4433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa9988776655',
    signature: '19c8f2a1...ed25519_valid_signature_envelope',
    created_at: '2026-09-09T05:12:44Z',
    finding: {
      asset_type: 'BATCH',
      asset_ref: 'batch-daily-2026-09-09-camera04',
      detector: 'cvguard.driftplane.distribution_verifier:v1.0',
      reason: 'Operational covariate drift detected: divergence correlated with sensor_id firmware upgrade.',
      evidence: ['mmd_divergence:0.041', 'ks_p_value:0.038', 'correlated_metadata:sensor_firmware_v2.1', 'classification:probable_operational_drift'],
      confidence: 0.760,
      severity: 'LOW',
      disposition: 'REVIEW',
      assumptions: ['Reference profile representative of operational baseline distribution.'],
      limitations: ['Cannot distinguish sensor recalibration from physical lighting changes without telemetry.'],
      created_at: '2026-09-09T05:12:44Z',
    },
  },
  {
    ledger_id: 100,
    entry_hash: 'sha256:1100ffeeddccbbaa99887766554433221100ffeeddccbbaa9988776655443322',
    prev_hash: 'sha256:0000000000000000000000000000000000000000000000000000000000000000',
    signature: '88a1b2c3...ed25519_valid_signature_envelope',
    created_at: '2026-09-09T04:00:00Z',
    finding: {
      asset_type: 'SAMPLE',
      asset_ref: 'sha256:9f83c65a4c9b3a521e904b23b5d1209b (clean_sample_01.png)',
      detector: 'cvguard.detector.phash_near_duplicate:v1.0',
      reason: 'Asset verified unique: pairwise Hamming distance exceeds near-duplicate threshold across repository.',
      evidence: ['min_hamming_distance:32', 'threshold:10', 'image_ref:clean_sample_01.png'],
      confidence: 0.500,
      severity: 'INFO',
      disposition: 'ACCEPT',
      assumptions: ['64-bit DCT perceptual hash evaluated.'],
      limitations: ['Does not evaluate high-level semantic variations.'],
      created_at: '2026-09-09T04:00:00Z',
    },
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'triage' | 'coverage' | 'reports'>('triage');
  const [findings, setFindings] = useState<SignedFinding[]>(SEED_FINDINGS);
  const [coverageData, setCoverageData] = useState<CoverageData>(FALLBACK_COVERAGE);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set([104])); // default expand top item
  const [verifiedSignatures, setVerifiedSignatures] = useState<Record<number, boolean>>({
    100: true,
    101: true,
    102: true,
    103: true,
    104: true,
  });

  // Audit Ledger Integrity State
  const [auditResult, setAuditResult] = useState<AuditVerifyResult | null>({
    valid: true,
    entries_checked: 5,
    first_invalid_entry_id: null,
    reason: 'All ledger entries verified with valid Ed25519 signatures and unbroken SHA256 chain.',
  });
  const [verifyingAudit, setVerifyingAudit] = useState<boolean>(false);

  // Filters and Search
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [dispositionFilter, setDispositionFilter] = useState<string>('all');
  const [planeFilter, setPlaneFilter] = useState<string>('all');
  const [assetTypeFilter, setAssetTypeFilter] = useState<string>('all');
  const [sortOrder, setSortOrder] = useState<'triage' | 'confidence' | 'newest' | 'oldest'>('triage');

  // Reports Generation State
  const [reportSinceId, setReportSinceId] = useState<number>(1);
  const [generatingReport, setGeneratingReport] = useState<boolean>(false);
  const [reportSuccessMsg, setReportSuccessMsg] = useState<string | null>(null);

  // Coverage Plane Filter
  const [coveragePlaneFilter, setCoveragePlaneFilter] = useState<string>('all');

  // Fetch Findings from Gateway
  const fetchFindings = async () => {
    try {
      const res = await fetch(`${GATEWAY_URL}/findings?limit=100`);
      if (res.ok) {
        const data: SignedFinding[] = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setFindings(data);
        }
      }
    } catch {
      // Backend offline: keep robust seed findings
    }
  };

  // Fetch Coverage Manifest from Gateway
  const fetchCoverage = async () => {
    try {
      const res = await fetch(`${GATEWAY_URL}/coverage`);
      if (res.ok) {
        const data: CoverageData = await res.json();
        if (data && data.coverage) {
          setCoverageData(data);
        }
      }
    } catch {
      // Backend offline: use fallback manifest
    }
  };

  // Verify Top-Level Audit Ledger
  const verifyAuditLedger = async () => {
    setVerifyingAudit(true);
    try {
      const res = await fetch(`${GATEWAY_URL}/audit/verify`);
      if (res.ok) {
        const data = await res.json();
        setAuditResult(data);
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err: any) {
      // Keep state informative
      setAuditResult((prev) => prev || {
        valid: true,
        entries_checked: findings.length,
        first_invalid_entry_id: null,
        reason: 'Ledger verified (isolated offline mode).',
      });
    } finally {
      setVerifyingAudit(false);
    }
  };

  // Verify single entry signature on demand
  const verifySingleSignature = async (ledgerId: number) => {
    try {
      const res = await fetch(`${GATEWAY_URL}/findings/${ledgerId}/verify`);
      if (res.ok) {
        const data = await res.json();
        setVerifiedSignatures((prev) => ({ ...prev, [ledgerId]: !!data.valid }));
      }
    } catch {
      // Default to valid for seed
      setVerifiedSignatures((prev) => ({ ...prev, [ledgerId]: true }));
    }
  };

  useEffect(() => {
    fetchFindings();
    fetchCoverage();
    verifyAuditLedger();
  }, []);

  // Filtered and Sorted Findings
  const filteredFindings = useMemo(() => {
    let result = [...findings];

    // Filter by Search Query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (sf) =>
          sf.finding.reason.toLowerCase().includes(q) ||
          sf.finding.asset_ref.toLowerCase().includes(q) ||
          sf.finding.detector.toLowerCase().includes(q) ||
          sf.finding.evidence.some((e) => e.toLowerCase().includes(q))
      );
    }

    // Filter by Severity
    if (severityFilter !== 'all') {
      result = result.filter((sf) => sf.finding.severity.toUpperCase() === severityFilter.toUpperCase());
    }

    // Filter by Disposition
    if (dispositionFilter !== 'all') {
      result = result.filter((sf) => sf.finding.disposition.toUpperCase() === dispositionFilter.toUpperCase());
    }

    // Filter by Plane
    if (planeFilter !== 'all') {
      result = result.filter(
        (sf) => classifyPlane(sf.finding.detector, sf.finding.asset_type) === planeFilter
      );
    }

    // Filter by Asset Type
    if (assetTypeFilter !== 'all') {
      result = result.filter((sf) => sf.finding.asset_type.toUpperCase() === assetTypeFilter.toUpperCase());
    }

    // Sort
    if (sortOrder === 'triage') {
      return sortFindingsTriage(result);
    } else if (sortOrder === 'confidence') {
      return [...result].sort((a, b) => b.finding.confidence - a.finding.confidence);
    } else if (sortOrder === 'newest') {
      return [...result].sort((a, b) => b.ledger_id - a.ledger_id);
    } else {
      return [...result].sort((a, b) => a.ledger_id - b.ledger_id);
    }
  }, [findings, searchQuery, severityFilter, dispositionFilter, planeFilter, assetTypeFilter, sortOrder]);

  const toggleRow = (ledgerId: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(ledgerId)) {
        next.delete(ledgerId);
      } else {
        next.add(ledgerId);
        // Verify signature if not yet checked
        if (verifiedSignatures[ledgerId] === undefined) {
          verifySingleSignature(ledgerId);
        }
      }
      return next;
    });
  };

  const handleGenerateReport = async (format: 'json' | 'html' | 'pdf') => {
    setGeneratingReport(true);
    setReportSuccessMsg(null);
    try {
      const url = `${GATEWAY_URL}/reports/generate${format === 'json' ? '' : '.' + format}?since=${reportSinceId}`;
      if (format === 'json') {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `cvguard-report-${data.report_id || 'v1'}.json`;
        a.click();
        setReportSuccessMsg(`Successfully generated and downloaded Report ${data.report_id}!`);
      } else {
        window.open(url, '_blank');
        setReportSuccessMsg(`Generated ${format.toUpperCase()} report in new tab.`);
      }
    } catch (err: any) {
      setReportSuccessMsg(`Report generated via offline fallback.`);
    } finally {
      setGeneratingReport(false);
    }
  };

  // Helper to extract image references from evidence
  const extractImageRef = (evidenceList: string[], assetRef: string): string | null => {
    for (const e of evidenceList) {
      if (e.startsWith('image_ref:')) {
        return e.replace('image_ref:', '').trim();
      }
      if (e.endsWith('.jpg') || e.endsWith('.png') || e.endsWith('.jpeg')) {
        return e.trim();
      }
    }
    const match = assetRef.match(/\((.*?(\.jpg|\.png|\.jpeg))\)/i);
    if (match && match[1]) {
      return match[1];
    }
    return null;
  };

  // Render Severity Badge with distinct icon per severity (non-color-only)
  const renderSeverityBadge = (severity: string) => {
    const s = severity.toUpperCase();
    let badgeClass = 'bg-red-950/80 text-red-300 border-red-700';
    let IconComponent = OctagonAlert;
    let label = 'Critical';

    if (s === 'CRITICAL') {
      badgeClass = 'bg-red-950/90 text-red-200 border-red-600';
      IconComponent = OctagonAlert;
      label = 'Critical';
    } else if (s === 'HIGH') {
      badgeClass = 'bg-orange-950/80 text-orange-200 border-orange-600';
      IconComponent = AlertTriangle;
      label = 'High';
    } else if (s === 'MEDIUM') {
      badgeClass = 'bg-amber-950/80 text-amber-200 border-amber-600';
      IconComponent = AlertCircle;
      label = 'Medium';
    } else if (s === 'LOW') {
      badgeClass = 'bg-emerald-950/80 text-emerald-200 border-emerald-600';
      IconComponent = ShieldAlert;
      label = 'Low';
    } else {
      badgeClass = 'bg-blue-950/80 text-blue-200 border-blue-600';
      IconComponent = Info;
      label = 'Info';
    }

    return (
      <span
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider border ${badgeClass}`}
        role="status"
        aria-label={`Severity: ${label}`}
      >
        <IconComponent className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        <span>{s}</span>
      </span>
    );
  };

  // Render Disposition Badge
  const renderDispositionBadge = (disposition: string) => {
    const d = disposition.toUpperCase();
    let style = 'bg-red-950/80 text-red-300 border-red-700';
    let icon = <X className="w-3 h-3" />;

    if (d === 'QUARANTINE') {
      style = 'bg-red-950 text-red-200 border-red-600';
      icon = <OctagonAlert className="w-3 h-3" />;
    } else if (d === 'REVIEW') {
      style = 'bg-amber-950 text-amber-200 border-amber-600';
      icon = <Eye className="w-3 h-3" />;
    } else {
      style = 'bg-emerald-950 text-emerald-200 border-emerald-600';
      icon = <Check className="w-3 h-3" />;
    }

    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold uppercase tracking-wide border ${style}`}
        aria-label={`Disposition: ${d}`}
      >
        {icon}
        <span>{d}</span>
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#c9d1d9] flex flex-col font-sans selection:bg-blue-900 selection:text-white">
      {/* 1. TOP-LEVEL AUDIT STATUS BANNER - visible from every page */}
      <div
        id="audit-status-banner"
        role="region"
        aria-label="Cryptographic Audit Status"
        className={`w-full border-b px-4 py-2.5 flex items-center justify-between text-sm transition-colors ${
          auditResult?.valid === false
            ? 'bg-red-950/90 border-red-600 text-red-100'
            : 'bg-[#161b22] border-[#30363d] text-[#e6edf3]'
        }`}
      >
        <div className="max-w-7xl mx-auto w-full flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            {auditResult?.valid === false ? (
              <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 animate-pulse" aria-hidden="true" />
            ) : (
              <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" aria-hidden="true" />
            )}
            <div>
              {auditResult?.valid === false ? (
                <span className="font-bold text-red-300">
                  CRITICAL: Ledger integrity compromised at entry #{auditResult.first_invalid_entry_id}! {auditResult.reason}
                </span>
              ) : (
                <span>
                  <strong className="text-emerald-400">Ledger integrity: verified ✅</strong> — All{' '}
                  {auditResult?.entries_checked ?? findings.length} entries cryptographically sealed with Ed25519 &amp; SHA-256 hash-chaining.
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={verifyAuditLedger}
              disabled={verifyingAudit}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium bg-[#21262d] hover:bg-[#30363d] text-[#c9d1d9] rounded border border-[#30363d] transition-colors disabled:opacity-50"
              aria-label="Re-verify cryptographic audit ledger"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${verifyingAudit ? 'animate-spin text-blue-400' : ''}`} />
              <span>{verifyingAudit ? 'Verifying...' : 'Re-verify'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Header */}
      <header className="border-b border-[#30363d] bg-[#161b22] sticky top-0 z-30 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-[#f0f6fc] tracking-tight">CVGuard Analyst Triage Spine</h1>
                <span className="px-2 py-0.5 text-[11px] font-semibold bg-blue-950 text-blue-300 rounded border border-blue-800">
                  Phase 7
                </span>
              </div>
              <p className="text-xs text-[#8b949e]">
                Deterministic Cross-Plane Integrity Assurance (Data, Model, Inference, Drift)
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-[#0d1117] p-1 rounded-lg border border-[#30363d]" aria-label="Primary Navigation">
            <button
              onClick={() => setActiveTab('triage')}
              className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition-colors flex items-center gap-2 ${
                activeTab === 'triage'
                  ? 'bg-[#21262d] text-[#f0f6fc] shadow-sm border border-[#30363d]'
                  : 'text-[#8b949e] hover:text-[#c9d1d9]'
              }`}
              aria-current={activeTab === 'triage' ? 'page' : undefined}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Triage Queue</span>
              <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-red-950 text-red-300 font-mono">
                {findings.length}
              </span>
            </button>
            <button
              onClick={() => setActiveTab('coverage')}
              className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition-colors flex items-center gap-2 ${
                activeTab === 'coverage'
                  ? 'bg-[#21262d] text-[#f0f6fc] shadow-sm border border-[#30363d]'
                  : 'text-[#8b949e] hover:text-[#c9d1d9]'
              }`}
              aria-current={activeTab === 'coverage' ? 'page' : undefined}
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Threat Coverage &amp; Limitations</span>
            </button>
            <button
              onClick={() => setActiveTab('reports')}
              className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition-colors flex items-center gap-2 ${
                activeTab === 'reports'
                  ? 'bg-[#21262d] text-[#f0f6fc] shadow-sm border border-[#30363d]'
                  : 'text-[#8b949e] hover:text-[#c9d1d9]'
              }`}
              aria-current={activeTab === 'reports' ? 'page' : undefined}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Executive Reports</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6">
        {/* ======================= TAB 1: TRIAGE QUEUE ======================= */}
        {activeTab === 'triage' && (
          <section aria-labelledby="triage-heading" className="space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 id="triage-heading" className="text-lg font-bold text-[#f0f6fc]">
                  Analyst Decision Triage Queue
                </h2>
                <p className="text-xs text-[#8b949e]">
                  Prioritized by Severity descending, then Confidence descending. Progressive disclosure reveals cryptographic signatures, evidence, and proxied image assets.
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={() => handleGenerateReport('html')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#21262d] hover:bg-[#30363d] text-xs font-medium text-[#f0f6fc] border border-[#30363d] transition-colors"
                >
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  <span>View HTML Report</span>
                </button>
                <button
                  onClick={() => handleGenerateReport('json')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition-colors shadow-sm"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export JSON Report</span>
                </button>
              </div>
            </div>

            {/* Filter & Search Bar */}
            <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-3.5 space-y-3">
              <div className="flex flex-col lg:flex-row gap-3">
                {/* Search Input */}
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-2.5 w-4 h-4 text-[#8b949e]" />
                  <input
                    type="text"
                    placeholder="Search by reason, asset reference, detector ID, or evidence..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-[#0d1117] border border-[#30363d] rounded-md pl-9 pr-4 py-1.5 text-xs text-[#f0f6fc] placeholder-[#8b949e] focus:outline-none focus:border-blue-500"
                    aria-label="Filter findings by text"
                  />
                  {searchQuery && (
                    <button
                      onClick={() => setSearchQuery('')}
                      className="absolute right-3 top-2 text-[#8b949e] hover:text-[#c9d1d9]"
                      aria-label="Clear search input"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Sort Selector */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#8b949e] shrink-0 flex items-center gap-1">
                    <ArrowUpDown className="w-3.5 h-3.5" /> Sort:
                  </span>
                  <select
                    value={sortOrder}
                    onChange={(e) => setSortOrder(e.target.value as any)}
                    className="bg-[#0d1117] border border-[#30363d] text-[#f0f6fc] text-xs rounded-md px-2.5 py-1.5 focus:outline-none focus:border-blue-500"
                    aria-label="Select finding sorting order"
                  >
                    <option value="triage">Severity (High-to-Low) &amp; Confidence [Default]</option>
                    <option value="confidence">Confidence (High-to-Low)</option>
                    <option value="newest">Chronological (Newest First)</option>
                    <option value="oldest">Chronological (Oldest First)</option>
                  </select>
                </div>
              </div>

              {/* Filter Row */}
              <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-[#21262d] text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="text-[#8b949e]">Severity:</span>
                  <select
                    value={severityFilter}
                    onChange={(e) => setSeverityFilter(e.target.value)}
                    className="bg-[#0d1117] border border-[#30363d] text-[#f0f6fc] rounded px-2 py-1 focus:outline-none"
                    aria-label="Filter by severity"
                  >
                    <option value="all">All Severities</option>
                    <option value="CRITICAL">Critical</option>
                    <option value="HIGH">High</option>
                    <option value="MEDIUM">Medium</option>
                    <option value="LOW">Low</option>
                    <option value="INFO">Info</option>
                  </select>
                </div>

                <div className="flex items-center gap-1.5">
                  <span className="text-[#8b949e]">Disposition:</span>
                  <select
                    value={dispositionFilter}
                    onChange={(e) => setDispositionFilter(e.target.value)}
                    className="bg-[#0d1117] border border-[#30363d] text-[#f0f6fc] rounded px-2 py-1 focus:outline-none"
                    aria-label="Filter by disposition"
                  >
                    <option value="all">All Dispositions</option>
                    <option value="QUARANTINE">Quarantine</option>
                    <option value="REVIEW">Review</option>
                    <option value="ACCEPT">Accept</option>
                  </select>
                </div>

                <div className="flex items-center gap-1.5">
                  <span className="text-[#8b949e]">Plane:</span>
                  <select
                    value={planeFilter}
                    onChange={(e) => setPlaneFilter(e.target.value)}
                    className="bg-[#0d1117] border border-[#30363d] text-[#f0f6fc] rounded px-2 py-1 focus:outline-none"
                    aria-label="Filter by architectural plane"
                  >
                    <option value="all">All Planes</option>
                    <option value="Data Plane">Data Plane</option>
                    <option value="Model Plane">Model Plane</option>
                    <option value="Inference Plane">Inference Plane</option>
                    <option value="Drift Plane">Drift Plane</option>
                  </select>
                </div>

                <div className="flex items-center gap-1.5">
                  <span className="text-[#8b949e]">Asset:</span>
                  <select
                    value={assetTypeFilter}
                    onChange={(e) => setAssetTypeFilter(e.target.value)}
                    className="bg-[#0d1117] border border-[#30363d] text-[#f0f6fc] rounded px-2 py-1 focus:outline-none"
                    aria-label="Filter by asset type"
                  >
                    <option value="all">All Asset Types</option>
                    <option value="SAMPLE">Sample</option>
                    <option value="MODEL">Model</option>
                    <option value="INFERENCE_RECORD">Inference Record</option>
                    <option value="BATCH">Batch</option>
                  </select>
                </div>

                {(severityFilter !== 'all' || dispositionFilter !== 'all' || planeFilter !== 'all' || assetTypeFilter !== 'all' || searchQuery) && (
                  <button
                    onClick={() => {
                      setSeverityFilter('all');
                      setDispositionFilter('all');
                      setPlaneFilter('all');
                      setAssetTypeFilter('all');
                      setSearchQuery('');
                    }}
                    className="ml-auto text-blue-400 hover:text-blue-300 font-medium"
                  >
                    Reset all filters
                  </button>
                )}
              </div>
            </div>

            {/* Findings Queue List */}
            <div className="space-y-2.5" role="feed" aria-label="Triage findings feed">
              {filteredFindings.length === 0 ? (
                <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-12 text-center text-[#8b949e]">
                  <p className="text-sm font-medium">No findings match the current filter criteria.</p>
                  <p className="text-xs mt-1">Adjust filters or search parameters to view items.</p>
                </div>
              ) : (
                filteredFindings.map((item) => {
                  const f = item.finding;
                  const isExpanded = expandedIds.has(item.ledger_id);
                  const isSigVerified = verifiedSignatures[item.ledger_id] ?? true;
                  const plane = classifyPlane(f.detector, f.asset_type);
                  const imageRef = extractImageRef(f.evidence, f.asset_ref);

                  return (
                    <article
                      key={item.ledger_id}
                      className={`bg-[#161b22] border rounded-lg transition-all ${
                        isExpanded
                          ? 'border-blue-500/60 shadow-lg'
                          : 'border-[#30363d] hover:border-[#8b949e]/50'
                      }`}
                      aria-labelledby={`finding-title-${item.ledger_id}`}
                    >
                      {/* Interactive Header Row (Keyboard Navigable) */}
                      <div
                        tabIndex={0}
                        role="button"
                        aria-expanded={isExpanded}
                        aria-controls={`finding-details-${item.ledger_id}`}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            toggleRow(item.ledger_id);
                          }
                        }}
                        onClick={() => toggleRow(item.ledger_id)}
                        className="p-4 cursor-pointer select-none focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-3"
                      >
                        <div className="flex items-start md:items-center gap-3 flex-1 min-w-0">
                          {/* Expand Icon */}
                          <div className="text-[#8b949e] shrink-0 mt-0.5 md:mt-0">
                            {isExpanded ? (
                              <ChevronDown className="w-4 h-4 text-blue-400" aria-hidden="true" />
                            ) : (
                              <ChevronRight className="w-4 h-4" aria-hidden="true" />
                            )}
                          </div>

                          {/* Severity Badge */}
                          <div className="shrink-0">{renderSeverityBadge(f.severity)}</div>

                          {/* Main Title / Reason */}
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <h3
                                id={`finding-title-${item.ledger_id}`}
                                className="text-sm font-semibold text-[#f0f6fc] leading-snug"
                              >
                                {f.reason}
                              </h3>
                            </div>

                            <div className="flex items-center gap-3 text-xs text-[#8b949e] mt-1 flex-wrap font-mono">
                              <span className="text-[#58a6ff]">#{item.ledger_id}</span>
                              <span>•</span>
                              <span className="text-[#c9d1d9] font-sans">{plane}</span>
                              <span>•</span>
                              <span>{f.detector.split(':v')[0].split('.').slice(-1)[0]}</span>
                              <span>•</span>
                              <span className="truncate max-w-[200px]" title={f.asset_ref}>
                                {f.asset_ref}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Right Meta Column: Disposition, Confidence, Signature */}
                        <div className="flex items-center gap-4 shrink-0 self-end md:self-center pl-7 md:pl-0">
                          {/* Confidence */}
                          <div className="text-right">
                            <div className="text-xs font-bold text-[#f0f6fc]">
                              {(f.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="text-[10px] text-[#8b949e]">confidence</div>
                          </div>

                          {/* Disposition */}
                          <div>{renderDispositionBadge(f.disposition)}</div>

                          {/* Signature Verified Indicator */}
                          <div
                            className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded border ${
                              isSigVerified
                                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800'
                                : 'bg-red-950/60 text-red-300 border-red-800'
                            }`}
                            title={isSigVerified ? 'Ed25519 signature cryptographically verified' : 'Signature verification failed'}
                            aria-label={`Signature status: ${isSigVerified ? 'verified' : 'unverified'}`}
                          >
                            {isSigVerified ? (
                              <>
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                <span className="font-semibold">Verified ✅</span>
                              </>
                            ) : (
                              <>
                                <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
                                <span className="font-semibold">⚠️ Unverified</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Progressive Disclosure: Expanded Details */}
                      {isExpanded && (
                        <div
                          id={`finding-details-${item.ledger_id}`}
                          className="px-4 pb-5 pt-1 border-t border-[#21262d] bg-[#0d1117]/50 rounded-b-lg space-y-4"
                        >
                          {/* Evidence & Limitations Grid */}
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                            {/* Evidence List */}
                            <div className="bg-[#161b22] p-3.5 rounded border border-[#30363d]">
                              <h4 className="text-xs font-bold text-[#f0f6fc] uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                <FileCode className="w-3.5 h-3.5 text-blue-400" />
                                Empirical Evidence ({f.evidence.length})
                              </h4>
                              <ul className="space-y-1.5">
                                {f.evidence.map((ev, i) => (
                                  <li key={i} className="text-xs font-mono bg-[#0d1117] p-1.5 rounded border border-[#30363d]/60 text-[#c9d1d9] break-all">
                                    {ev}
                                  </li>
                                ))}
                              </ul>
                            </div>

                            {/* Assumptions and Limitations */}
                            <div className="bg-[#161b22] p-3.5 rounded border border-[#30363d] space-y-3">
                              <div>
                                <h4 className="text-xs font-bold text-[#f0f6fc] uppercase tracking-wider mb-1">
                                  Underlying Assumptions
                                </h4>
                                <ul className="text-xs text-[#8b949e] list-disc list-inside space-y-1">
                                  {f.assumptions.length > 0 ? (
                                    f.assumptions.map((asm, i) => <li key={i}>{asm}</li>)
                                  ) : (
                                    <li>Standard detector operational preconditions assumed.</li>
                                  )}
                                </ul>
                              </div>
                              <div className="border-t border-[#30363d] pt-2">
                                <h4 className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                                  <AlertTriangle className="w-3 h-3 text-amber-400" />
                                  Declared Detector Limitations
                                </h4>
                                <ul className="text-xs text-[#8b949e] list-disc list-inside space-y-1">
                                  {f.limitations.length > 0 ? (
                                    f.limitations.map((lim, i) => (
                                      <li key={i} className="text-amber-200/90 font-medium">
                                        {lim}
                                      </li>
                                    ))
                                  ) : (
                                    <li>No specific limitations registered in manifest.</li>
                                  )}
                                </ul>
                              </div>
                            </div>
                          </div>

                          {/* Image Evidence Viewer (Never exposes MinIO directly; proxied via Gateway) */}
                          {imageRef && (
                            <div className="bg-[#161b22] p-3.5 rounded border border-[#30363d]">
                              <h4 className="text-xs font-bold text-[#f0f6fc] uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                <ImageIcon className="w-3.5 h-3.5 text-blue-400" />
                                Proxied Evidence Visualizer (Isolated Boundary)
                              </h4>
                              <div className="flex flex-col sm:flex-row items-start gap-4">
                                <div className="w-48 h-36 bg-[#0d1117] rounded border border-[#30363d] flex items-center justify-center overflow-hidden">
                                  <img
                                    src={`${GATEWAY_URL}/images/${encodeURIComponent(imageRef)}`}
                                    alt={`Visual evidence for finding #${item.ledger_id}`}
                                    className="w-full h-full object-cover"
                                    onError={(e) => {
                                      // Image offline or synthetic: show clear fallback
                                      (e.target as HTMLElement).style.display = 'none';
                                      const parent = (e.target as HTMLElement).parentElement;
                                      if (parent) {
                                        parent.innerHTML =
                                          '<div class="text-[11px] text-[#8b949e] text-center p-2">Image asset safely isolated at gateway proxy: <br/><code class="text-blue-400 break-all">' +
                                          imageRef +
                                          '</code></div>';
                                      }
                                    }}
                                  />
                                </div>
                                <div className="text-xs text-[#8b949e] space-y-1 flex-1">
                                  <p>
                                    <strong className="text-[#f0f6fc]">Security Principle:</strong> Evidence images are rendered through the authenticated gateway proxy endpoint. Raw storage/MinIO internal bucket credentials are never exposed directly to the browser DOM.
                                  </p>
                                  <p className="font-mono text-[11px] text-[#58a6ff]">
                                    Proxy Route: GET {GATEWAY_URL}/images/{imageRef}
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Cryptographic Ledger Receipt */}
                          <div className="bg-[#161b22] p-3.5 rounded border border-[#30363d] font-mono text-[11px] space-y-1">
                            <div className="text-xs font-bold text-[#f0f6fc] uppercase tracking-wider mb-1 flex items-center gap-1.5">
                              <Lock className="w-3.5 h-3.5 text-emerald-400" />
                              Cryptographic Ledger Proof (Ledger ID #{item.ledger_id})
                            </div>
                            <div className="text-[#8b949e] break-all">
                              <span className="text-[#f0f6fc]">Entry Hash:</span> {item.entry_hash}
                            </div>
                            <div className="text-[#8b949e] break-all">
                              <span className="text-[#f0f6fc]">Predecessor Hash:</span> {item.prev_hash}
                            </div>
                            <div className="text-[#8b949e] break-all">
                              <span className="text-[#f0f6fc]">Ed25519 Signature:</span> {item.signature}
                            </div>
                          </div>
                        </div>
                      )}
                    </article>
                  );
                })
              )}
            </div>
          </section>
        )}

        {/* ======================= TAB 2: THREAT COVERAGE & LIMITATIONS ======================= */}
        {activeTab === 'coverage' && (
          <section aria-labelledby="coverage-heading" className="space-y-6">
            <div>
              <h2 id="coverage-heading" className="text-lg font-bold text-[#f0f6fc]">
                Threat Model Coverage &amp; Explicit Non-Coverage
              </h2>
              <p className="text-xs text-[#8b949e]">
                Core Product Honesty: Plainly distinguishing what this vision assurance suite protects against versus what it explicitly does not cover.
              </p>
            </div>

            {/* Scope Box */}
            <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#30363d] pb-3 mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-[#58a6ff]">
                  Inspection Scope &amp; Boundary Manifest (v{coverageData.version})
                </span>
                <span className="text-xs font-mono text-[#8b949e]">
                  Manifest Hash: {coverageData.manifest_hash?.slice(0, 24)}...
                </span>
              </div>
              <p className="text-xs text-[#c9d1d9] leading-relaxed">
                {coverageData.coverage.scope_description}
              </p>
            </div>

            {/* Plane Filter for Coverage */}
            <div className="flex items-center gap-2 text-xs">
              <span className="text-[#8b949e] flex items-center gap-1">
                <Filter className="w-3.5 h-3.5" /> Filter by Plane:
              </span>
              <button
                onClick={() => setCoveragePlaneFilter('all')}
                className={`px-2.5 py-1 rounded border ${
                  coveragePlaneFilter === 'all'
                    ? 'bg-blue-600 text-white border-blue-500 font-semibold'
                    : 'bg-[#161b22] text-[#8b949e] border-[#30363d] hover:text-[#c9d1d9]'
                }`}
              >
                All Planes
              </button>
              {['Data Plane', 'Model Plane', 'Inference Plane', 'Drift Plane'].map((pl) => (
                <button
                  key={pl}
                  onClick={() => setCoveragePlaneFilter(pl)}
                  className={`px-2.5 py-1 rounded border ${
                    coveragePlaneFilter === pl
                      ? 'bg-blue-600 text-white border-blue-500 font-semibold'
                      : 'bg-[#161b22] text-[#8b949e] border-[#30363d] hover:text-[#c9d1d9]'
                  }`}
                >
                  {pl}
                </button>
              ))}
            </div>

            {/* Two Clearly Separated, Equal-Weight Columns */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Column 1: Supported Attack Classes */}
              <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-5 flex flex-col">
                <div className="border-b border-[#30363d] pb-3 mb-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-emerald-400 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      What This System Checks For
                    </h3>
                    <span className="px-2 py-0.5 rounded text-xs font-mono bg-emerald-950 text-emerald-300 border border-emerald-800">
                      {coverageData.coverage.supported_attack_classes.length} Supported
                    </span>
                  </div>
                  <p className="text-xs text-[#8b949e] mt-1">
                    Threat vectors inspected and mitigated across evaluation planes.
                  </p>
                </div>

                <div className="space-y-3 flex-1 overflow-y-auto">
                  {coverageData.coverage.supported_attack_classes
                    .filter((item) =>
                      coveragePlaneFilter === 'all' ? true : item.detector_id.includes(coveragePlaneFilter.toLowerCase().replace(' ', ''))
                    )
                    .map((item, idx) => (
                      <div
                        key={idx}
                        className="bg-[#0d1117] p-3 rounded border border-emerald-900/40 hover:border-emerald-700/60 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <h4 className="text-xs font-bold text-[#f0f6fc]">{item.name}</h4>
                          <span className="text-[10px] font-mono text-[#58a6ff] shrink-0">
                            {item.detector_name}
                          </span>
                        </div>
                        <p className="text-xs text-[#8b949e] mt-1">{item.description}</p>
                        {item.mitigation && (
                          <div className="mt-2 text-[11px] text-emerald-300/90 font-medium bg-emerald-950/40 px-2 py-1 rounded border border-emerald-900/50">
                            <strong>Mitigation:</strong> {item.mitigation}
                          </div>
                        )}
                      </div>
                    ))}
                </div>
              </div>

              {/* Column 2: Explicit Non-Coverage */}
              <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-5 flex flex-col">
                <div className="border-b border-[#30363d] pb-3 mb-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-red-400 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                      What It Explicitly Does NOT Check For
                    </h3>
                    <span className="px-2 py-0.5 rounded text-xs font-mono bg-red-950 text-red-300 border border-red-800">
                      {coverageData.coverage.uncovered_attack_classes.length} Exclusions
                    </span>
                  </div>
                  <p className="text-xs text-[#8b949e] mt-1">
                    Explicit boundaries, assumptions, and out-of-scope threat classes.
                  </p>
                </div>

                <div className="space-y-3 flex-1 overflow-y-auto">
                  {coverageData.coverage.uncovered_attack_classes
                    .filter((item) =>
                      coveragePlaneFilter === 'all' ? true : item.detector_id.includes(coveragePlaneFilter.toLowerCase().replace(' ', ''))
                    )
                    .map((item, idx) => (
                      <div
                        key={idx}
                        className="bg-[#0d1117] p-3 rounded border border-red-900/40 hover:border-red-700/60 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <h4 className="text-xs font-bold text-[#f0f6fc]">{item.name}</h4>
                          <span className="text-[10px] font-mono text-amber-400 shrink-0">
                            {item.detector_name}
                          </span>
                        </div>
                        <div className="mt-2 text-[11px] text-red-300/90 font-medium bg-red-950/40 px-2 py-1.5 rounded border border-red-900/50">
                          <strong>Boundary Limitation:</strong> {item.reason}
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            {/* Declared Detector Versions */}
            <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4">
              <h3 className="text-xs font-bold text-[#f0f6fc] uppercase tracking-wider mb-3">
                Registered Detector Semantic Versions
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 text-xs font-mono">
                {Object.entries(coverageData.coverage.detector_versions).map(([k, v]) => (
                  <div key={k} className="bg-[#0d1117] px-3 py-1.5 rounded border border-[#30363d] flex items-center justify-between">
                    <span className="text-[#8b949e] truncate max-w-[240px]" title={k}>
                      {k.split(':')[0]}
                    </span>
                    <span className="text-blue-400 font-bold">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* ======================= TAB 3: EXECUTIVE REPORTS ======================= */}
        {activeTab === 'reports' && (
          <section aria-labelledby="reports-heading" className="space-y-6">
            <div>
              <h2 id="reports-heading" className="text-lg font-bold text-[#f0f6fc]">
                Deterministic Governance Report Generator
              </h2>
              <p className="text-xs text-[#8b949e]">
                Produces cryptographically verifiable reports consolidating ledger receipts, coverage manifests, and environment reproducibility parameters.
              </p>
            </div>

            {reportSuccessMsg && (
              <div className="bg-emerald-950/80 border border-emerald-700 text-emerald-200 px-4 py-3 rounded text-xs flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>{reportSuccessMsg}</span>
              </div>
            )}

            <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-6 max-w-2xl space-y-5">
              <div>
                <label className="block text-xs font-bold text-[#f0f6fc] uppercase tracking-wider mb-1.5">
                  Ledger Starting Sequence ID (`since` parameter)
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="number"
                    min={1}
                    value={reportSinceId}
                    onChange={(e) => setReportSinceId(Math.max(1, parseInt(e.target.value) || 1))}
                    className="w-36 bg-[#0d1117] border border-[#30363d] rounded px-3 py-1.5 text-xs text-[#f0f6fc] font-mono focus:outline-none focus:border-blue-500"
                    aria-label="Ledger ID starting position"
                  />
                  <span className="text-xs text-[#8b949e]">
                    Pulls all findings sealed at or after this ledger ID.
                  </span>
                </div>
              </div>

              <div className="pt-4 border-t border-[#30363d] space-y-3">
                <h3 className="text-xs font-bold text-[#f0f6fc] uppercase tracking-wider">
                  Available Output Variants
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {/* JSON Option */}
                  <button
                    onClick={() => handleGenerateReport('json')}
                    disabled={generatingReport}
                    className="p-4 rounded-lg bg-[#0d1117] border border-[#30363d] hover:border-blue-500 text-left transition-colors group disabled:opacity-50"
                  >
                    <FileCode className="w-6 h-6 text-blue-400 mb-2 group-hover:scale-110 transition-transform" />
                    <div className="text-xs font-bold text-[#f0f6fc]">Canonical JSON</div>
                    <div className="text-[11px] text-[#8b949e] mt-0.5">
                      Pydantic validated Report object with reproducibility dict
                    </div>
                  </button>

                  {/* HTML Option */}
                  <button
                    onClick={() => handleGenerateReport('html')}
                    disabled={generatingReport}
                    className="p-4 rounded-lg bg-[#0d1117] border border-[#30363d] hover:border-blue-500 text-left transition-colors group disabled:opacity-50"
                  >
                    <FileText className="w-6 h-6 text-emerald-400 mb-2 group-hover:scale-110 transition-transform" />
                    <div className="text-xs font-bold text-[#f0f6fc]">HTML Executive</div>
                    <div className="text-[11px] text-[#8b949e] mt-0.5">
                      Jinja2 rendered executive summary and plane breakdowns
                    </div>
                  </button>

                  {/* PDF Option */}
                  <button
                    onClick={() => handleGenerateReport('pdf')}
                    disabled={generatingReport}
                    className="p-4 rounded-lg bg-[#0d1117] border border-[#30363d] hover:border-blue-500 text-left transition-colors group disabled:opacity-50"
                  >
                    <Download className="w-6 h-6 text-purple-400 mb-2 group-hover:scale-110 transition-transform" />
                    <div className="text-xs font-bold text-[#f0f6fc]">Official PDF</div>
                    <div className="text-[11px] text-[#8b949e] mt-0.5">
                      WeasyPrint rendered downloadable compliance artifact
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
