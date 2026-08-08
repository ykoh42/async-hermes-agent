import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/quickstart',
        'getting-started/installation',
      ],
    },
    {
      type: 'category',
      label: 'Using Async Hermes Agent',
      collapsed: true,
      items: [
        'user-guide/configuration',
        'user-guide/configuring-models',
        'user-guide/sessions',
        'user-guide/security',
      ],
    },
    {
      type: 'category',
      label: 'Features',
      collapsed: true,
      items: [
        'user-guide/features/overview',
        'user-guide/features/tools',
        'user-guide/features/skills',
        'user-guide/features/mcp',
        'user-guide/features/memory',
        'user-guide/features/browser',
        'user-guide/features/batch-processing',
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      collapsed: true,
      items: [
        'integrations/index',
        'integrations/providers',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: true,
      items: [
        'guides/python-library',
        'guides/use-mcp-with-hermes',
        'guides/work-with-skills',
      ],
    },
    {
      type: 'category',
      label: 'Developer Guide',
      collapsed: true,
      items: [
        'developer-guide/architecture',
        'developer-guide/agent-loop',
        'developer-guide/programmatic-integration',
        'developer-guide/session-storage',
        'developer-guide/trajectory-format',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: true,
      items: [
        'reference/environment-variables',
        'reference/mcp-config-reference',
        'reference/tools-reference',
        'reference/toolsets-reference',
        'reference/faq',
      ],
    },
  ],
};

export default sidebars;
