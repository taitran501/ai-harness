import React, { useMemo, useCallback, useState, useRef, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  Node,
  Edge,
  MarkerType,
  useReactFlow,
  useUpdateNodeInternals,
  ReactFlowProvider,
  NodeTypes,
} from '@xyflow/react';
import dagre from 'dagre';
import { ArrowDown, ArrowRight, RefreshCw, GitBranch } from 'lucide-react';
import { CustomExperimentNode } from './CustomExperimentNode';
import { TreeGraphResponse, GraphNode, GraphEdge, UCBRecommendation } from '../types/experiment';
import { useLanguage } from '../constants/i18n';

interface TreeCanvasProps {
  graphData: TreeGraphResponse;
  onSelectNode: (nodeId: string) => void;
  selectedNodeId: string | null;
  recommendation?: UCBRecommendation | null;
}

const nodeTypes: NodeTypes = {
  experimentNode: CustomExperimentNode,
};

const NODE_WIDTH = 240;
const NODE_HEIGHT = 80;

// Compute hierarchical layout with Dagre
function getLayoutedElements(
  nodes: GraphNode[],
  edges: GraphEdge[],
  direction: 'TB' | 'LR' = 'TB'
): { layoutedNodes: Node[]; layoutedEdges: Edge[] } {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  const isLR = direction === 'LR';
  dagreGraph.setGraph({
    rankdir: direction,
    nodesep: isLR ? 28 : 36,
    ranksep: isLR ? 92 : 56,
    marginx: 24,
    marginy: 24,
  });

  nodes.forEach((n) => {
    dagreGraph.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  edges.forEach((e) => {
    dagreGraph.setEdge(e.from, e.to);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes: Node[] = nodes.map((n) => {
    const nodeWithPos = dagreGraph.node(n.id) || { x: 0, y: 0 };
    return {
      id: n.id,
      type: 'experimentNode',
      position: {
        x: nodeWithPos.x - NODE_WIDTH / 2,
        y: nodeWithPos.y - NODE_HEIGHT / 2,
      },
      data: {
        ...n,
        layoutDirection: direction,
      },
    };
  });

  const layoutedEdges: Edge[] = edges.map((e, index) => {
    const hasDelta = typeof e.delta === 'number' && !isNaN(e.delta);
    // Epsilon threshold to prevent -0.0000 or +0.0000 floating point display artifacts
    const isPositive = hasDelta && e.delta >= 0.00005;
    const isNegative = hasDelta && e.delta <= -0.00005;

    const strokeColor = isPositive ? '#10b981' : isNegative ? '#ef4444' : '#64748b';
    const textColor = isPositive ? '#34d399' : isNegative ? '#f87171' : '#94a3b8';
    const borderStroke = isPositive ? '#059669' : isNegative ? '#b91c1c' : '#475569';
    const labelText = hasDelta
      ? (isPositive ? `+${e.delta.toFixed(4)}` : isNegative ? e.delta.toFixed(4) : '0.0000')
      : '0.0000';

    return {
      id: `e-${e.from}-${e.to}-${index}`,
      source: e.from,
      target: e.to,
      type: 'smoothstep',
      animated: false,
      label: labelText,
      labelStyle: {
        fill: textColor,
        fontFamily: 'JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
        fontSize: 10,
        fontWeight: 600,
      },
      labelBgStyle: {
        fill: '#090d16',
        fillOpacity: 0.95,
        stroke: borderStroke,
        strokeWidth: 1,
      },
      labelBgPadding: [5, 2],
      labelBgBorderRadius: 4,
      style: {
        stroke: strokeColor,
        strokeWidth: 1.5,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: strokeColor,
        width: 12,
        height: 12,
      },
    };
  });

  return { layoutedNodes, layoutedEdges };
}

const FlowInner: React.FC<TreeCanvasProps> = ({ graphData, onSelectNode, selectedNodeId, recommendation }) => {
  const { t } = useLanguage();
  const [direction, setDirection] = useState<'TB' | 'LR'>('TB');
  const { fitView } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  // Deduplicate and filter out dummy template nodes
  const filteredNodes = useMemo(() => {
    const rawNodes = graphData?.nodes || [];
    const seen = new Set<string>();
    const unique: GraphNode[] = [];
    for (const n of rawNodes) {
      if (n && typeof n.id === 'string' && !n.id.includes('template') && !seen.has(n.id)) {
        seen.add(n.id);
        unique.push(n);
      }
    }
    return unique;
  }, [graphData?.nodes]);

  const filteredNodeIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes]);

  // Filter valid directed edges without self-loops or orphaned endpoints
  const filteredEdges = useMemo(() => {
    const rawEdges = graphData?.edges || [];
    const seenKeys = new Set<string>();
    const uniqueEdges: typeof rawEdges = [];
    for (const e of rawEdges) {
      if (!e || !e.from || !e.to || e.from === e.to) continue;
      if (!filteredNodeIds.has(e.from) || !filteredNodeIds.has(e.to)) continue;
      const key = `${e.from}->${e.to}`;
      if (!seenKeys.has(key)) {
        seenKeys.add(key);
        uniqueEdges.push(e);
      }
    }
    return uniqueEdges;
  }, [graphData?.edges, filteredNodeIds]);

  const { layoutedNodes, layoutedEdges } = useMemo(() => {
    return getLayoutedElements(filteredNodes, filteredEdges, direction);
  }, [filteredNodes, filteredEdges, direction]);

  // Set selected state
  const nodes = useMemo(() => {
    return layoutedNodes.map((n) => ({
      ...n,
      selected: n.id === selectedNodeId,
    }));
  }, [layoutedNodes, selectedNodeId]);

  // Auto-fit view when solution tree is first populated from empty state
  const prevCountRef = useRef(filteredNodes.length);
  useEffect(() => {
    if (prevCountRef.current === 0 && filteredNodes.length > 0) {
      requestAnimationFrame(() => {
        setTimeout(() => {
          fitView({ duration: 400, padding: 0.2 });
        }, 50);
      });
    }
    prevCountRef.current = filteredNodes.length;
  }, [filteredNodes.length, fitView]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      onSelectNode(node.id);
    },
    [onSelectNode]
  );

  const handleToggleDirection = (dir: 'TB' | 'LR') => {
    if (dir === direction) return;
    setDirection(dir);
    // Update node internals after DOM updates to sync handle anchors accurately
    requestAnimationFrame(() => {
      updateNodeInternals(filteredNodes.map((n) => n.id));
      setTimeout(() => {
        fitView({ duration: 400, padding: 0.2 });
      }, 50);
    });
  };

  const handleRecenter = useCallback(() => {
    fitView({ duration: 400, padding: 0.2 });
  }, [fitView]);

  return (
    <div className="w-full h-[calc(100vh-140px)] min-h-[500px] rounded-xl overflow-hidden border border-zinc-800/80 bg-zinc-950 relative flex flex-col">
      {/* Top Floating Controls */}
      <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-1 pointer-events-auto bg-zinc-900/90 backdrop-blur-sm border border-zinc-800 p-0.5 rounded-lg text-xs">
          <button
            type="button"
            title={t('tree_tb_tooltip', 'Top-Down layout')}
            onClick={() => handleToggleDirection('TB')}
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md transition ${
              direction === 'TB'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <ArrowDown className="w-3 h-3" />
            <span>{t('tree_tb', 'Top-Bottom')}</span>
          </button>
          <button
            type="button"
            title={t('tree_lr_tooltip', 'Left-Right layout')}
            onClick={() => handleToggleDirection('LR')}
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md transition ${
              direction === 'LR'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <ArrowRight className="w-3 h-3" />
            <span>{t('tree_lr', 'Left-Right')}</span>
          </button>
        </div>

        <div className="pointer-events-auto">
          <button
            type="button"
            title={t('tree_fit_tooltip', 'Fit graph in view')}
            onClick={handleRecenter}
            className="flex items-center gap-1.5 bg-zinc-900/90 hover:bg-zinc-800 active:scale-95 backdrop-blur-sm border border-zinc-800 px-2.5 py-1 rounded-lg text-xs font-medium text-zinc-300 transition"
          >
            <RefreshCw className="w-3 h-3 text-zinc-400" />
            <span>{t('tree_recenter', 'Recenter')}</span>
          </button>

        </div>
      </div>

      {/* Empty State Overlay */}
      {filteredNodes.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-500 pointer-events-none z-10 space-y-1.5">
          <GitBranch className="w-7 h-7 opacity-20 text-zinc-400" />
          <p className="text-xs font-medium text-zinc-400">{t('tree_empty_title', 'No experiment nodes in solution tree')}</p>
          <p className="text-[11px] text-zinc-600">{t('tree_empty_desc', 'Run an experiment or click Sync to scan disk artifacts')}</p>
        </div>
      )}

      {/* Canvas */}
      <div className="flex-1 w-full h-full">
        <ReactFlow
          nodes={nodes}
          edges={layoutedEdges}
          nodeTypes={nodeTypes}
          onNodeClick={onNodeClick}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={true}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.2}
          maxZoom={1.5}
        >
          <Background color="#27272a" gap={24} size={1} />
          <Controls className="!m-3" showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
};

export const TreeCanvas: React.FC<TreeCanvasProps> = (props) => {
  return (
    <ReactFlowProvider>
      <FlowInner {...props} />
    </ReactFlowProvider>
  );
};

