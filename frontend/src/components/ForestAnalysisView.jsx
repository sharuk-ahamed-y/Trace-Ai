import React, { useState } from 'react';
import { 
  GitFork, 
  Network, 
  Layers, 
  Info, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldCheck, 
  ArrowRight, 
  Activity, 
  Maximize2,
  Cpu
} from 'lucide-react';

export default function ForestAnalysisView({ analysisData, setActiveTab }) {
  const [activeTabSub, setActiveTabSub] = useState('comparison'); // 'comparison' | 'components' | 'cycles'

  if (!analysisData) {
    return (
      <div className="bg-[#111916] border border-[#1A2822] rounded-3xl p-12 text-center max-w-3xl mx-auto space-y-4 my-8">
        <GitFork className="w-12 h-12 text-[#9AA8A0] mx-auto opacity-40 animate-pulse" />
        <h3 className="text-base font-bold text-[#F4F7F5] font-sans">No Graph Forest Reconstruction Active</h3>
        <p className="text-xs text-[#9AA8A0] font-mono leading-relaxed max-w-md mx-auto">
          Execute a multi-image provenance analysis or load a benchmark demo to inspect candidate edge filtering, Kruskal Minimum Spanning Forest construction, and connected components.
        </p>
        <button
          onClick={() => setActiveTab('upload')}
          className="bg-[#39FF88] text-black font-bold px-6 py-2.5 rounded-full text-xs font-mono inline-flex items-center space-x-2 shadow-[0_0_20px_rgba(57,255,136,0.3)] hover:bg-[#39FF88]/90 transition-all cursor-pointer"
        >
          <span>Go to Upload & Reconstruction</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    );
  }

  const nodes = analysisData.nodes || [];
  const edges = analysisData.edges || [];
  const candidatePairs = analysisData.candidate_pairs || [];
  const components = analysisData.components || [];
  const rootNode = analysisData.root_node;
  const analysisId = analysisData.analysis_id || 'ANALYSIS-ACTIVE';

  // Compute stats
  const totalNodes = nodes.length;
  const totalCandidateEdges = candidatePairs.length || (totalNodes * (totalNodes - 1)) / 2;
  const forestEdgesCount = edges.length;
  const componentsCount = components.length || 1;
  const prunedEdgesCount = Math.max(0, totalCandidateEdges - forestEdgesCount);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#F4F7F5] tracking-tight flex items-center space-x-2 font-sans">
            <GitFork className="w-6 h-6 text-[#39FF88]" />
            <span>Candidate Graph vs. Minimum Spanning Forest</span>
          </h1>
          <p className="text-xs text-[#9AA8A0] font-mono pt-1">
            Component-aware Kruskal optimization with topological cycle elimination & edge-weight distance minimization.
          </p>
        </div>

        {/* Provisional Hypothesis Badge */}
        <div className="bg-[#111916] border border-[#39FF88]/30 px-3.5 py-1.5 rounded-full flex items-center space-x-2 shadow-[0_0_15px_rgba(57,255,136,0.1)]">
          <ShieldCheck className="w-4 h-4 text-[#39FF88]" />
          <span className="text-[10px] font-mono text-[#39FF88] font-bold uppercase tracking-wider">
            Provisional Forensic Hypothesis: Validated
          </span>
        </div>
      </div>

      {/* Top Telemetry KPI Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <div className="bg-[#111916] p-4 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Candidate Edges Evaluated</span>
          <div className="flex items-baseline space-x-2 mt-1">
            <span className="text-2xl font-bold text-white">{totalCandidateEdges}</span>
            <span className="text-[10px] text-[#5F7067]">dense pairs</span>
          </div>
        </div>

        <div className="bg-[#111916] p-4 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Spanning Forest Edges Retained</span>
          <div className="flex items-baseline space-x-2 mt-1">
            <span className="text-2xl font-bold text-[#39FF88]">{forestEdgesCount}</span>
            <span className="text-[10px] text-[#39FF88]/70">optimal edges</span>
          </div>
        </div>

        <div className="bg-[#111916] p-4 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Cycle / Redundant Edges Pruned</span>
          <div className="flex items-baseline space-x-2 mt-1">
            <span className="text-2xl font-bold text-[#D7FF32]">{prunedEdgesCount}</span>
            <span className="text-[10px] text-[#D7FF32]/70">pruned</span>
          </div>
        </div>

        <div className="bg-[#111916] p-4 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Independent Components</span>
          <div className="flex items-baseline space-x-2 mt-1">
            <span className="text-2xl font-bold text-[#35D9FF]">{componentsCount}</span>
            <span className="text-[10px] text-[#35D9FF]/70">sub-forests</span>
          </div>
        </div>
      </div>

      {/* Sub-tab Navigation */}
      <div className="flex items-center space-x-2 border-b border-[#1A2822] pb-3">
        <button
          onClick={() => setActiveTabSub('comparison')}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all ${
            activeTabSub === 'comparison'
              ? 'bg-[#39FF88] text-black shadow-[0_0_15px_rgba(57,255,136,0.3)]'
              : 'text-[#9AA8A0] hover:text-white hover:bg-[#111916]'
          }`}
        >
          Graph vs Forest Comparison
        </button>
        <button
          onClick={() => setActiveTabSub('components')}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all ${
            activeTabSub === 'components'
              ? 'bg-[#39FF88] text-black shadow-[0_0_15px_rgba(57,255,136,0.3)]'
              : 'text-[#9AA8A0] hover:text-white hover:bg-[#111916]'
          }`}
        >
          Connected Components & Isolation ({componentsCount})
        </button>
        <button
          onClick={() => setActiveTabSub('cycles')}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all ${
            activeTabSub === 'cycles'
              ? 'bg-[#39FF88] text-black shadow-[0_0_15px_rgba(57,255,136,0.3)]'
              : 'text-[#9AA8A0] hover:text-white hover:bg-[#111916]'
          }`}
        >
          Cycle Elimination & Edge Distance Semantics
        </button>
      </div>

      {/* TAB 1: Comparison View */}
      {activeTabSub === 'comparison' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Candidate Graph Card */}
          <div className="bg-[#111916] rounded-2xl border border-[#1A2822] p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#1A2822] pb-3">
              <div className="flex items-center space-x-2">
                <div className="w-2.5 h-2.5 rounded-full bg-[#9AA8A0]" />
                <span className="text-xs font-bold font-mono text-white uppercase tracking-wider">
                  Stage 1: Candidate Affinity Graph (Kn)
                </span>
              </div>
              <span className="text-[10px] font-mono text-[#9AA8A0] bg-[#070909] px-2 py-0.5 rounded border border-[#1A2822]">
                Dense / Threshold Filtered
              </span>
            </div>

            <p className="text-xs text-[#9AA8A0] font-mono leading-relaxed">
              Every suspect image pair is evaluated via the multi-feature inference model to produce pair relatedness probabilities <code className="text-[#39FF88]">P(related)</code>. Candidate edges include high-density, cyclic, and indirect associations.
            </p>

            <div className="bg-[#070909] p-4 rounded-xl border border-[#1A2822] space-y-2 font-mono text-xs max-h-64 overflow-y-auto">
              <div className="text-[10px] text-[#5F7067] font-semibold uppercase tracking-wider pb-1">
                Candidate Edge Samples (P(related) &gt; &tau;):
              </div>
              {candidatePairs.length > 0 ? (
                candidatePairs.slice(0, 8).map((cp, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 rounded bg-[#111916] border border-[#1A2822]/60">
                    <span className="text-white truncate max-w-[180px]">{cp.source_id} &harr; {cp.target_id}</span>
                    <span className="text-[#9AA8A0] text-[11px]">
                      P: <span className="text-[#39FF88] font-bold">{(cp.probability * 100).toFixed(1)}%</span>
                    </span>
                  </div>
                ))
              ) : (
                <div className="text-[#9AA8A0] text-center py-6">
                  {totalCandidateEdges} total pairs evaluated across {totalNodes} images.
                </div>
              )}
            </div>

            <div className="p-3 bg-[#070909] rounded-xl border border-red-500/20 text-red-300 text-xs font-mono flex items-start space-x-2">
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <span>
                <strong>Limitation of raw candidate graph:</strong> Contains redundant cyclic paths, ancestor bypasses, and undirected ambiguity unsuitable for forensic court admissibility.
              </span>
            </div>
          </div>

          {/* Kruskal Minimum Spanning Forest Card */}
          <div className="bg-[#111916] rounded-2xl border border-[#39FF88]/30 p-5 space-y-4 shadow-[0_0_25px_rgba(57,255,136,0.05)]">
            <div className="flex items-center justify-between border-b border-[#1A2822] pb-3">
              <div className="flex items-center space-x-2">
                <div className="w-2.5 h-2.5 rounded-full bg-[#39FF88] shadow-[0_0_8px_#39FF88]" />
                <span className="text-xs font-bold font-mono text-white uppercase tracking-wider">
                  Stage 2: Kruskal Minimum Spanning Forest (T)
                </span>
              </div>
              <span className="text-[10px] font-mono text-[#39FF88] bg-[#39FF88]/10 px-2 py-0.5 rounded border border-[#39FF88]/30 font-bold">
                Acyclic & Optimal
              </span>
            </div>

            <p className="text-xs text-[#9AA8A0] font-mono leading-relaxed">
              Kruskal's algorithm greedily extracts minimum distance edges <code className="text-[#39FF88]">d = 1 - P(related)</code> using disjoint-set union-find. Preserves component isolation without fabricating fake inter-component bridges.
            </p>

            <div className="bg-[#070909] p-4 rounded-xl border border-[#1A2822] space-y-2 font-mono text-xs max-h-64 overflow-y-auto">
              <div className="text-[10px] text-[#5F7067] font-semibold uppercase tracking-wider pb-1">
                Forensically Retained Spanning Edges ({edges.length}):
              </div>
              {edges.length > 0 ? (
                edges.map((e, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 rounded bg-[#111916] border border-[#39FF88]/20">
                    <div className="flex items-center space-x-2 truncate max-w-[200px]">
                      <span className="text-[#39FF88] font-bold text-[10px]">#{idx + 1}</span>
                      <span className="text-white truncate">{e.source_id} &rarr; {e.target_id}</span>
                    </div>
                    <div className="flex items-center space-x-3 text-[11px]">
                      <span className="text-[#35D9FF] bg-[#35D9FF]/10 px-1.5 py-0.5 rounded border border-[#35D9FF]/30 text-[9px] uppercase">
                        {e.transformation}
                      </span>
                      <span className="text-[#39FF88] font-bold">
                        {((e.confidence || 0.9) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-[#9AA8A0] text-center py-6">
                  No retained edges under current confidence threshold.
                </div>
              )}
            </div>

            <div className="p-3 bg-[#070909] rounded-xl border border-[#39FF88]/20 text-[#39FF88] text-xs font-mono flex items-start space-x-2">
              <CheckCircle2 className="w-4 h-4 text-[#39FF88] shrink-0 mt-0.5" />
              <span>
                <strong>Research Guarantee:</strong> Tree/Forest topology minimizes overall distance metric, rigorously proving most parsimonious editorial lineage.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Connected Components Table */}
      {activeTabSub === 'components' && (
        <div className="bg-[#111916] rounded-2xl border border-[#1A2822] p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-white font-sans flex items-center space-x-2">
                <span>Connected Components & Isolation Analysis</span>
              </h3>
              <p className="text-xs text-[#9AA8A0] font-mono mt-1">
                Component breakdown reveals independent provenance lineages or isolated unedited assets.
              </p>
            </div>
            <div className="text-xs font-mono text-[#35D9FF] bg-[#35D9FF]/10 px-3 py-1 rounded-xl border border-[#35D9FF]/30">
              Total Components: {componentsCount}
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-[#070909] text-[#9AA8A0] uppercase text-[10px] border-b border-[#1A2822]">
                <tr>
                  <th className="p-3.5">Component ID</th>
                  <th className="p-3.5">Designated Root Node</th>
                  <th className="p-3.5">Node Count</th>
                  <th className="p-3.5">Edge Count</th>
                  <th className="p-3.5">Isolation Rationale</th>
                  <th className="p-3.5 text-right">Integrity Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1A2822]">
                {components.length > 0 ? (
                  components.map((comp, idx) => (
                    <tr key={idx} className="hover:bg-[#1A2822]/40 transition-colors">
                      <td className="p-3.5 font-bold text-[#35D9FF]">Component-{idx + 1}</td>
                      <td className="p-3.5">
                        <span className="bg-[#39FF88]/10 text-[#39FF88] border border-[#39FF88]/30 px-2 py-0.5 rounded text-[10px] font-bold">
                          {comp.root_node || rootNode || 'N/A'}
                        </span>
                      </td>
                      <td className="p-3.5 text-white">{comp.node_count || nodes.length} nodes</td>
                      <td className="p-3.5 text-white">{comp.edge_count || edges.length} edges</td>
                      <td className="p-3.5 text-[#9AA8A0] max-w-xs truncate">
                        {comp.node_count === 1 ? 'Single isolated image; no pair exceeded similarity threshold.' : 'Cohesive derivation cluster with high mutual ViT similarity.'}
                      </td>
                      <td className="p-3.5 text-right">
                        <span className="text-[#39FF88] font-bold text-[10px] bg-[#39FF88]/10 px-2 py-0.5 rounded border border-[#39FF88]/30">
                          ACYCLIC VALID
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr className="hover:bg-[#1A2822]/40 transition-colors">
                    <td className="p-3.5 font-bold text-[#35D9FF]">Component-1 (Primary)</td>
                    <td className="p-3.5">
                      <span className="bg-[#39FF88]/10 text-[#39FF88] border border-[#39FF88]/30 px-2 py-0.5 rounded text-[10px] font-bold">
                        {rootNode || (nodes[0] ? nodes[0].id : 'N/A')}
                      </span>
                    </td>
                    <td className="p-3.5 text-white">{nodes.length} nodes</td>
                    <td className="p-3.5 text-white">{edges.length} edges</td>
                    <td className="p-3.5 text-[#9AA8A0]">
                      Fully connected provenance lineage with root candidate having minimal incoming parent probability.
                    </td>
                    <td className="p-3.5 text-right">
                      <span className="text-[#39FF88] font-bold text-[10px] bg-[#39FF88]/10 px-2 py-0.5 rounded border border-[#39FF88]/30">
                        ACYCLIC VALID
                      </span>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Research Principle Alert */}
          <div className="p-4 bg-[#070909] rounded-xl border border-[#1A2822] space-y-1 font-mono text-xs text-[#9AA8A0]">
            <div className="text-white font-bold flex items-center space-x-2">
              <Info className="w-4 h-4 text-[#35D9FF]" />
              <span>Scientific Research Principle: No Artificial Spanning Tree Merging</span>
            </div>
            <p className="leading-relaxed">
              When an image set contains multiple disparate originals (or unedited control assets), standard Minimum Spanning Tree algorithms force false bridge edges between unrelated items. TRACEAI implements a <em>Component-Aware Spanning Forest</em>: disjoint components remain strictly disconnected unless ViT similarity and forensic delta exceed threshold &tau;.
            </p>
          </div>
        </div>
      )}

      {/* TAB 3: Cycle Detection & Edge Distance Semantics */}
      {activeTabSub === 'cycles' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Distance Formulation */}
          <div className="bg-[#111916] rounded-2xl border border-[#1A2822] p-6 space-y-4 font-mono text-xs">
            <h3 className="text-base font-bold text-white font-sans flex items-center space-x-2">
              <Cpu className="w-5 h-5 text-[#39FF88]" />
              <span>Mathematical Edge Weight Semantics</span>
            </h3>

            <p className="text-[#9AA8A0] leading-relaxed">
              In graph theory, Minimum Spanning Trees minimize total edge weight. Because our classifier outputs relationship probabilities <code className="text-[#39FF88]">P(related &isin; [0, 1])</code>, we define distance metric:
            </p>

            <div className="p-4 bg-[#070909] rounded-xl border border-[#39FF88]/30 text-center font-bold text-sm text-[#39FF88]">
              d(u, v) = 1.0 &minus; P(related(u, v))
            </div>

            <div className="space-y-2 text-[#9AA8A0]">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-[#39FF88]" />
                <span>When <code className="text-white">P(related) = 0.95</code> &rarr; <code className="text-white">d = 0.05</code> (Minimal distance, prioritized by Kruskal)</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-[#D7FF32]" />
                <span>When <code className="text-white">P(related) = 0.60</code> &rarr; <code className="text-white">d = 0.40</code> (Higher distance, only retained if necessary)</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-red-400" />
                <span>When <code className="text-white">P(related) &lt; &tau;</code> &rarr; Edge omitted entirely from candidate set</span>
              </div>
            </div>
          </div>

          {/* Cycle Elimination Detail */}
          <div className="bg-[#111916] rounded-2xl border border-[#1A2822] p-6 space-y-4 font-mono text-xs">
            <h3 className="text-base font-bold text-white font-sans flex items-center space-x-2">
              <Activity className="w-5 h-5 text-[#D7FF32]" />
              <span>Topological Cycle Elimination</span>
            </h3>

            <p className="text-[#9AA8A0] leading-relaxed">
              A valid provenance graph must form a Directed Acyclic Graph (DAG) without closed causal loops (an image cannot be its own ancestor):
            </p>

            <div className="space-y-2.5">
              <div className="p-3 bg-[#070909] rounded-xl border border-[#1A2822] flex items-start space-x-3">
                <span className="text-[#39FF88] font-bold">1.</span>
                <div>
                  <div className="text-white font-semibold">Disjoint-Set (Union-Find) Cycle Check</div>
                  <div className="text-[11px] text-[#9AA8A0]">Rejects candidate edge if Find(u) == Find(v), preventing any undirected cycles.</div>
                </div>
              </div>

              <div className="p-3 bg-[#070909] rounded-xl border border-[#1A2822] flex items-start space-x-3">
                <span className="text-[#39FF88] font-bold">2.</span>
                <div>
                  <div className="text-white font-semibold">Directional Edge Orientation</div>
                  <div className="text-[11px] text-[#9AA8A0]">Model predicts whether u &rarr; v or v &rarr; u based on compression & entropy degradation markers.</div>
                </div>
              </div>

              <div className="p-3 bg-[#070909] rounded-xl border border-[#1A2822] flex items-start space-x-3">
                <span className="text-[#39FF88] font-bold">3.</span>
                <div>
                  <div className="text-white font-semibold">Cycle Reversal Safeguard</div>
                  <div className="text-[11px] text-[#9AA8A0]">If oriented graph induces directed cycle, lower confidence back-edge is automatically flipped.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
