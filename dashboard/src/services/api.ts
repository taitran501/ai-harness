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
  LLMProviderInfo,
  LLMSampleDataResponse,
  LLMTestPromptRequest,
  LLMTestPromptResponse,
  LLMPeftGenerateRequest,
  LLMPeftGenerateResponse,
  HFModelInfo,
  UnslothModelInfo,
  ProviderModelInfo,
  LLMUnslothGenerateRequest,
  StrategyCard,
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

  async getPrimingPrompt(templateIdOrIde?: string, intent?: string): Promise<PrimingPromptResponse> {
    let query = '';
    if (intent && templateIdOrIde) {
      query = `?ide=${encodeURIComponent(templateIdOrIde)}&intent=${encodeURIComponent(intent)}`;
    } else if (templateIdOrIde) {
      query = `?template=${encodeURIComponent(templateIdOrIde)}`;
    }
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

  async createExperimentConfig(data: any): Promise<{ success: boolean; path: string; parsed: any }> {
    const res = await fetch(`${API_BASE}/config/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to create config: ${res.statusText}`);
    }
    return res.json();
  },

  async getConfig(
    path?: string,
    expId?: string
  ): Promise<{ path: string; content: string; parsed: any }> {
    const params = new URLSearchParams();
    if (path) params.append('path', path);
    if (expId) params.append('exp_id', expId);
    const res = await fetch(`${API_BASE}/config?${params.toString()}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to fetch config: ${res.statusText}`);
    }
    return res.json();
  },

  async saveConfig(
    path: string,
    content: string
  ): Promise<{ success: boolean; path: string; parsed: any }> {
    const res = await fetch(`${API_BASE}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to save config: ${res.statusText}`);
    }
    return res.json();
  },

  async getStrategyCards(): Promise<{ strategies: StrategyCard[] }> {
    const res = await fetch(`${API_BASE}/strategy_cards`);
    if (!res.ok) throw new Error(`Failed to fetch strategy cards: ${res.statusText}`);
    return res.json();
  },

  async createConfig(payload: {
    exp_id?: string;
    descriptor?: string;
    parent_id?: string;
    branch?: string;
    hypothesis?: string;
    intervention?: string;
    problem_addressed?: string;
    approach_type?: string;
    model_family?: string;
    model_name?: string;
    model_params?: Record<string, any>;
    param_size?: string | null;
    llm_spec?: any;
    source_type?: string;
    source_id?: string;
    retrieval_summary?: string;
    uncertainty_level?: string;
  }): Promise<{ success: boolean; path: string; content: string; parsed: any }> {
    const res = await fetch(`${API_BASE}/config/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to create config: ${res.statusText}`);
    }
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

  async getNextExpId(branch?: string, descriptor?: string): Promise<import('../types/experiment').NextExpIdResponse> {
    const params = new URLSearchParams();
    if (branch) params.append('branch', branch);
    if (descriptor) params.append('descriptor', descriptor);
    const res = await fetch(`${API_BASE}/next_exp_id?${params.toString()}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to fetch next experiment ID: ${res.statusText}`);
    }
    return res.json();
  },

  async getEnvironmentStatus(): Promise<import('../types/experiment').EnvironmentStatus> {
    const res = await fetch(`${API_BASE}/environment/status`);
    if (!res.ok) throw new Error(`Failed to fetch environment status: ${res.statusText}`);
    return res.json();
  },

  async activateModalProfile(profile: string): Promise<{ success: boolean; profile: string; output?: string; error?: string }> {
    const res = await fetch(`${API_BASE}/environment/modal/activate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to activate modal profile: ${res.statusText}`);
    }
    return res.json();
  },

  async setModalToken(tokenId: string, tokenSecret: string, profile?: string): Promise<{ success: boolean; output?: string; error?: string }> {
    const res = await fetch(`${API_BASE}/environment/modal/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token_id: tokenId, token_secret: tokenSecret, profile }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to set modal token: ${res.statusText}`);
    }
    return res.json();
  },

  async switchKaggleProfile(username: string, key: string, profileName?: string): Promise<{ success: boolean; username: string; profile_name: string }> {
    const res = await fetch(`${API_BASE}/environment/kaggle/profile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, key, profile_name: profileName }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to switch kaggle profile: ${res.statusText}`);
    }
    return res.json();
  },

  async launchModalAuth(): Promise<{ success: boolean; auth_url: string; message: string }> {
    const res = await fetch(`${API_BASE}/environment/modal/launch_auth`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to launch Modal authorization: ${res.statusText}`);
    }
    return res.json();
  },

  async launchKaggleAuth(): Promise<{ success: boolean; auth_url: string; message: string }> {
    const res = await fetch(`${API_BASE}/environment/kaggle/launch_auth`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to launch Kaggle authorization: ${res.statusText}`);
    }
    return res.json();
  },

  async getTasks(): Promise<{ tasks: import('../types/experiment').TaskSummary[] }> {
    const res = await fetch(`${API_BASE}/tasks`);
    if (!res.ok) throw new Error(`Failed to fetch tasks: ${res.statusText}`);
    return res.json();
  },

  async switchTask(taskId: string): Promise<{ success: boolean; task_id: string; active_task: any }> {
    const res = await fetch(`${API_BASE}/tasks/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: taskId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to switch task: ${res.statusText}`);
    }
    return res.json();
  },

  async createNewTask(taskSpec: any): Promise<{ success: boolean; task_id: string; task: any }> {
    const res = await fetch(`${API_BASE}/tasks/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task: taskSpec }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to create new task: ${res.statusText}`);
    }
    return res.json();
  },

  async runCloudJob(payload: {
    backend: 'modal' | 'kaggle' | 'local';
    config_path: string;
    gpu?: string;
    auto_audit?: boolean;
    auto_sync?: boolean;
    baseline_csv?: string | null;
  }): Promise<{ success: boolean; job: import('../types/experiment').CloudJob }> {
    const res = await fetch(`${API_BASE}/cloud/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to launch cloud job: ${res.statusText}`);
    }
    return res.json();
  },

  async stopCloudJob(jobId: string): Promise<{ success: boolean; status?: string; error?: string }> {
    const res = await fetch(`${API_BASE}/cloud/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to stop job: ${res.statusText}`);
    }
    return res.json();
  },

  async getCloudJob(jobId: string): Promise<{ job: import('../types/experiment').CloudJob }> {
    const res = await fetch(`${API_BASE}/cloud/job?job_id=${encodeURIComponent(jobId)}`);
    if (!res.ok) throw new Error(`Failed to fetch cloud job: ${res.statusText}`);
    return res.json();
  },

  async getCloudJobs(): Promise<{ jobs: import('../types/experiment').CloudJob[] }> {
    const res = await fetch(`${API_BASE}/cloud/jobs`);
    if (!res.ok) throw new Error(`Failed to list cloud jobs: ${res.statusText}`);
    return res.json();
  },

  async runCloudAudit(
    expId?: string,
    candidateCsv?: string,
    baselineCsv?: string
  ): Promise<{
    success: boolean;
    passed: boolean;
    stdout: string;
    stderr: string;
    receipt: Record<string, any> | null;
  }> {
    const res = await fetch(`${API_BASE}/cloud/audit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        exp_id: expId,
        candidate_csv: candidateCsv,
        baseline_csv: baselineCsv,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Audit execution failed: ${res.statusText}`);
    }
    return res.json();
  },

  async getLLMProviders(): Promise<{ providers: LLMProviderInfo[] }> {
    const res = await fetch(`${API_BASE}/llm/providers`);
    if (!res.ok) throw new Error(`Failed to fetch LLM providers: ${res.statusText}`);
    return res.json();
  },

  async getLLMSampleData(): Promise<LLMSampleDataResponse> {
    const res = await fetch(`${API_BASE}/llm/sample_data`);
    if (!res.ok) throw new Error(`Failed to fetch sample data: ${res.statusText}`);
    return res.json();
  },

  async testLLMPrompt(payload: LLMTestPromptRequest): Promise<LLMTestPromptResponse> {
    const res = await fetch(`${API_BASE}/llm/test_prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `LLM test prompt failed: ${res.statusText}`);
    }
    return res.json();
  },

  async generateLLMPeftScript(payload: LLMPeftGenerateRequest): Promise<LLMPeftGenerateResponse> {
    const res = await fetch(`${API_BASE}/llm/generate_peft_script`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `PEFT script generation failed: ${res.statusText}`);
    }
    return res.json();
  },

  async saveLLMKey(keyName: string, keyValue: string): Promise<{ success: boolean; key_name: string }> {
    const res = await fetch(`${API_BASE}/environment/llm/key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key_name: keyName, key_value: keyValue }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to save API key: ${res.statusText}`);
    }
    return res.json();
  },

  async verifyLLMKey(
    providerOrKey: string,
    keyValue?: string
  ): Promise<{
    success: boolean;
    valid: boolean;
    provider: string;
    latency_ms?: number;
    message?: string;
    error?: string;
  }> {
    const res = await fetch(`${API_BASE}/environment/llm/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: providerOrKey, key_value: keyValue }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Failed to verify API key: ${res.statusText}`);
    }
    return res.json();
  },

  async getHFModels(params?: {
    search?: string;
    pipeline?: string;
    author?: string;
    family?: string;
    param_range?: string;
    quantization?: string;
    limit?: number;
    sort?: string;
    force_refresh?: boolean;
  }): Promise<{ models: HFModelInfo[] }> {
    const query = new URLSearchParams();
    if (params?.search) query.append('search', params.search);
    if (params?.pipeline) query.append('pipeline', params.pipeline);
    if (params?.author) query.append('author', params.author);
    if (params?.family && params.family !== 'all') query.append('family', params.family);
    if (params?.param_range && params.param_range !== 'all') query.append('param_range', params.param_range);
    if (params?.quantization && params.quantization !== 'all') query.append('quantization', params.quantization);
    if (params?.limit) query.append('limit', params.limit.toString());
    if (params?.sort) query.append('sort', params.sort);
    if (params?.force_refresh) query.append('force_refresh', 'true');
    const res = await fetch(`${API_BASE}/llm/hf_models?${query.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch Hugging Face models: ${res.statusText}`);
    return res.json();
  },

  async getUnslothModels(params?: {
    search?: string;
    family?: string;
    param_range?: string;
    quantization?: string;
    limit?: number;
    sort?: string;
    force_refresh?: boolean;
  }): Promise<{ models: UnslothModelInfo[] }> {
    const query = new URLSearchParams();
    if (params?.search) query.append('search', params.search);
    if (params?.family && params.family !== 'all') query.append('family', params.family);
    if (params?.param_range && params.param_range !== 'all') query.append('param_range', params.param_range);
    if (params?.quantization && params.quantization !== 'all') query.append('quantization', params.quantization);
    query.append('limit', (params?.limit || 50).toString());
    if (params?.sort) query.append('sort', params.sort);
    if (params?.force_refresh) query.append('force_refresh', 'true');
    const res = await fetch(`${API_BASE}/llm/unsloth_models?${query.toString()}`);
    if (!res.ok) throw new Error(`Failed to fetch Unsloth models: ${res.statusText}`);
    return res.json();
  },

  async getProviderModels(provider: string): Promise<{ provider: string; models: ProviderModelInfo[] }> {
    const res = await fetch(`${API_BASE}/llm/provider_models?provider=${encodeURIComponent(provider)}`);
    if (!res.ok) throw new Error(`Failed to fetch provider models: ${res.statusText}`);
    return res.json();
  },

  async generateUnslothScript(payload: LLMUnslothGenerateRequest): Promise<LLMPeftGenerateResponse> {
    const res = await fetch(`${API_BASE}/llm/generate_unsloth_script`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Unsloth script generation failed: ${res.statusText}`);
    }
    return res.json();
  },
};


