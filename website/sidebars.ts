import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/platform-support',
        'getting-started/updating',
        'getting-started/learning-path',
      ],
    },
    {
      type: 'category',
      label: 'Using Hermes',
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
        {
          type: 'category',
          label: 'Core',
          items: [
            'user-guide/features/tools',
            'user-guide/features/skills',
            'user-guide/features/memory',
          ],
        },
        {
          type: 'category',
          label: 'Automation',
          items: [
            'user-guide/features/batch-processing',
          ],
        },
        {
          type: 'category',
          label: 'Media & Web',
          items: [
            'user-guide/features/browser',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      collapsed: true,
      items: [
        'integrations/index',
        'integrations/providers',
        'user-guide/features/mcp',
      ],
    },
    {
      type: 'category',
      label: 'Guides & Tutorials',
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
        {
          type: 'category',
          label: 'Architecture',
          items: [
            'developer-guide/architecture',
            'developer-guide/upstream-differences',
            'developer-guide/agent-loop',
            'developer-guide/session-storage',
            'developer-guide/postgres-production-readiness',
          ],
        },
        {
          type: 'category',
          label: 'Extending',
          items: [
            'developer-guide/programmatic-integration',
          ],
        },
        {
          type: 'category',
          label: 'Internals',
          items: [
            'developer-guide/trajectory-format',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Configuration Reference',
          items: [
            'reference/environment-variables',
            'reference/mcp-config-reference',
          ],
        },
        {
          type: 'category',
          label: 'Tools & Skills Reference',
          items: [
            'reference/tools-reference',
            'reference/toolsets-reference',
          ],
        },
        'reference/faq',
      ],
    },
  ],
};

export default sidebars;
