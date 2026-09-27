import React, { createContext, useContext, useState } from 'react';

export type UILanguage = 'en' | 'vi';

export const UI_TRANSLATIONS: Record<UILanguage, Record<string, string>> = {
  en: {
    app_title: 'ML Experiment Harness',
    nav_experiments: 'Experiments',
    nav_tree: 'AIDE Tree',
    nav_compare: 'Compare',
    nav_llm_studio: 'LLM & Prompt Studio',
    nav_cloud_runner: 'Cloud Runner',
    nav_priming: 'Priming Hub',
    nav_submissions: 'Submissions',
    new_experiment: 'New Experiment',
    task_settings: 'Task Settings',
    refresh: 'Refresh',
    language_label: 'Language',
  },
  vi: {
    app_title: 'Hệ thống Quản lý Thực nghiệm ML',
    nav_experiments: 'Thực nghiệm',
    nav_tree: 'Cây tìm kiếm AIDE',
    nav_compare: 'So sánh',
    nav_llm_studio: 'LLM & Prompt Studio',
    nav_cloud_runner: 'Cloud Runner',
    nav_priming: 'Trung tâm Khởi tạo',
    nav_submissions: 'Nộp bài',
    new_experiment: 'Thực nghiệm mới',
    task_settings: 'Cấu hình Bài toán',
    refresh: 'Làm mới',
    language_label: 'Ngôn ngữ',
  },
};

interface LanguageContextValue {
  lang: UILanguage;
  setLang: (lang: UILanguage) => void;
  t: (key: string, fallback?: string) => string;
}

const LanguageContext = createContext<LanguageContextValue>({
  lang: 'en',
  setLang: () => {},
  t: (key: string, fallback?: string) => UI_TRANSLATIONS.en[key] || fallback || key,
});

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [lang, setLangState] = useState<UILanguage>(() => {
    const saved = localStorage.getItem('ml_harness_ui_lang');
    return saved === 'vi' ? 'vi' : 'en';
  });

  const setLang = (next: UILanguage) => {
    setLangState(next);
    localStorage.setItem('ml_harness_ui_lang', next);
  };

  const t = (key: string, fallback?: string): string => {
    return UI_TRANSLATIONS[lang]?.[key] || UI_TRANSLATIONS.en[key] || fallback || key;
  };

  return React.createElement(
    LanguageContext.Provider,
    { value: { lang, setLang, t } },
    children
  );
};

export const useLanguage = () => useContext(LanguageContext);
