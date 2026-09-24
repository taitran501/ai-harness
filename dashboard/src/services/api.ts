import {
  ExperimentSummary,
  NodeDetail,
  PrimingPromptResponse,
  PromptTemplateInfo,
  RegisterSubmissionPayload,
  SubmissionEntry,
  SystemStatus,
  TaskInfo,
  TaskPreset,
  TreeGraphResponse,
  UCBRecommendation,
} from '../types/experiment';

const API_BASE = '/api';

export const api = {
  async getTree(): Promise<TreeGraphResponse> {
    const res = await fetch(`${API_BASE}/tree`);
    if (!res.ok) throw new Error(`Failed to fetch tree: ${res.statusText}`);
    return res.json();
  },

  async getExperiments(): Promise<{ total: number; experiments: ExperimentSummary[] }> {
    const res = await fetch(`${API_BASE}/experiments`);
    if (!res.ok) throw new Error(`Failed to fetch experiments: ${res.statusText}`);
    return res.json();
  },

  async getNode(id: string): Promise<NodeDetail> {
    const res = await fetch(`${API_BASE}/node/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch node ${id}: ${res.statusText}`);
    return res.json();
  },

  async getRecommendation(maxParams?: number): Promise<UCBRecommendation> {
    const query = maxParams ? `?max_params=${maxParams}` : '';
    const res = await fetch(`${API_BASE}/recommend${query}`);
    if (!res.ok) throw new Error(`Failed to fetch recommendation: ${res.statusText}`);
    return res.json();
  },

  async getPromptTemplates(): Promise<{ templates: PromptTemplateInfo[] }> {
    const res = await fetch(`${API_BASE}/prompt_templates`);
    if (!res.ok) throw new Error(`Failed to fetch prompt templates: ${res.statusText}`);
    return res.json();
  },

  async getPrimingPrompt(templateId?: string): Promise<PrimingPromptResponse> {
    const query = templateId ? `?template=${encodeURIComponent(templateId)}` : '';
    const res = await fetch(`${API_BASE}/priming_prompt${query}`);
    if (!res.ok) throw new Error(`Failed to fetch priming prompt: ${res.statusText}`);
    return res.json();
  },

  async getStatus(): Promise<SystemStatus> {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`);
    return res.json();
  },

  async getTaskInfo(): Promise<TaskInfo> {
    const res = await fetch(`${API_BASE}/task_info`);
    if (!res.ok) throw new Error(`Failed to fetch task info: ${res.statusText}`);
    return res.json();
  },

  async getTaskPresets(): Promise<{ presets: TaskPreset[] }> {
    const res = await fetch(`${API_BASE}/task_presets`);
    if (!res.ok) throw new Error(`Failed to fetch task presets: ${res.statusText}`);
    return res.json();
  },

  async updateTaskInfo(taskData: Partial<TaskInfo>): Promise<{ success: boolean; task: TaskInfo }> {
    const res = await fetch(`${API_BASE}/task_info`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(taskData),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to update task config: ${res.statusText}`);
    }
    return res.json();
  },

  async registerSubmission(
    payload: RegisterSubmissionPayload
  ): Promise<{ success: boolean; submission: SubmissionEntry }> {
    const res = await fetch(`${API_BASE}/submissions/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to register submission: ${res.statusText}`);
    }
    return res.json();
  },

  async getSubmissions(): Promise<{ total: number; submissions: SubmissionEntry[] }> {
    const res = await fetch(`${API_BASE}/submissions`);
    if (!res.ok) throw new Error(`Failed to fetch submissions: ${res.statusText}`);
    return res.json();
  },

  async recordOfficialScore(
    subId: string,
    officialScores: Record<string, any>,
    notes: string = ''
  ): Promise<{ success: boolean; submission: SubmissionEntry }> {
    const res = await fetch(`${API_BASE}/submissions/record-score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sub_id: subId,
        official_scores: officialScores,
        notes,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to record score: ${res.statusText}`);
    }
    return res.json();
  },

  async syncArtifacts(): Promise<{
    success: boolean;
    synced_count: number;
    total_nodes: number;
  }> {
    const res = await fetch(`${API_BASE}/sync`);
    if (!res.ok) throw new Error(`Failed to sync artifacts: ${res.statusText}`);
    return res.json();
  },

  async getConfigs(): Promise<{ configs: string[] }> {
    const res = await fetch(`${API_BASE}/configs`);
    if (!res.ok) throw new Error(`Failed to fetch configs: ${res.statusText}`);
    return res.json();
  },

  async generateCloudRunner(payload: {
    backend: 'modal' | 'kaggle';
    config_path: string;
    gpu?: string;
    dataset_slug?: string;
    volume_name?: string;
  }): Promise<import('../types/experiment').CloudGenerateResult> {
    const res = await fetch(`${API_BASE}/cloud/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Cloud generation failed: ${res.statusText}`);
    }
    return res.json();
  },

  async runCloudPreflight(configPath: string): Promise<import('../types/experiment').CloudPreflightResult> {
    const query = `?config=${encodeURIComponent(configPath)}`;
    const res = await fetch(`${API_BASE}/cloud/preflight${query}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Pre-flight check failed: ${res.statusText}`);
    }
    return res.json();
  },
};
