/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                tech: {
                    bg: '#09090b',       // Zinc 950
                    card: '#18181b',     // Zinc 900
                    border: '#27272a',   // Zinc 800
                    text: '#e4e4e7',     // Zinc 200
                    muted: '#a1a1aa',    // Zinc 400
                    accent: '#8b5cf6',   // Violet 500
                }
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
            },
            typography: (theme) => ({
                DEFAULT: {
                    css: {
                        color: theme('colors.tech.text'),
                        a: {
                            color: theme('colors.tech.accent'),
                            '&:hover': {
                                color: '#a78bfa', // Violet 400
                            },
                        },
                        h1: { color: '#fff', fontWeight: '800', fontSize: '2.25em' },
                        h2: { color: '#fff', fontWeight: '700', fontSize: '1.75em', marginTop: '1.5em', marginBottom: '0.8em' },
                        h3: { color: '#e4e4e7', fontWeight: '600', fontSize: '1.375em', marginTop: '1.2em', marginBottom: '0.6em' },
                        h4: { color: '#e4e4e7', fontWeight: '600' },
                        strong: { color: '#fff' },
                        code: {
                            color: theme('colors.tech.accent'),
                            backgroundColor: 'rgba(139, 92, 246, 0.1)',
                            padding: '0.2em 0.4em',
                            borderRadius: '0.25rem',
                            fontWeight: '400',
                        },
                        'code::before': { content: '""' },
                        'code::after': { content: '""' },
                        blockquote: {
                            borderLeftColor: theme('colors.tech.accent'),
                            color: theme('colors.tech.muted'),
                        },
                        ul: {
                            listStyleType: 'disc',
                            paddingLeft: '1.5em',
                        },
                        li: {
                            marginTop: '0.5em',
                            marginBottom: '0.5em',
                        }
                    },
                },
            }),
        },
    },
    plugins: [
        require('@tailwindcss/typography'),
    ],
}
