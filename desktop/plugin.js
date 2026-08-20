/**
 * hermes-verifier — desktop half (unified package: <plugin dir>/desktop/plugin.js)
 *
 * Verifier Mode: a full page at /verifier styled like a normal session screen.
 * Header says "Verifier Mode"; underneath it sits the Verifier model selector
 * (the same model catalog the composer uses), an enable switch, and a live
 * status chip. The body is a session-style transcript of reviewed deltas and
 * findings; the bottom composer sends the verifier a direct question.
 *
 * The backend (dashboard/plugin_api.py) mounts at gateway startup; until then
 * the page degrades to an explanatory banner instead of a broken screen.
 */

import {
  host,
  haptic,
  useQuery,
  useQueryClient,
  ModelCatalogMenu,
  ModelMenuCloseContext,
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  Button,
  Switch,
  GlyphSpinner,
  ScrollArea,
  Separator,
  Textarea,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  PALETTE_AREA,
  relativeTime,
  cn,
} from '@hermes/plugin-sdk'
import { jsx, jsxs, Fragment } from 'react/jsx-runtime'
import { useState, useEffect, useCallback } from 'react'

const SEVERITY_STYLE = {
  note: { color: 'var(--ui-text-tertiary)', fontWeight: '500' },
  caution: { color: 'var(--ui-text-secondary)', fontWeight: '600' },
  blocker: { color: 'var(--ui-accent)', fontWeight: '700' },
}

function SeverityChip({ severity }) {
  if (!severity) return null
  const style = SEVERITY_STYLE[severity] || SEVERITY_STYLE.note
  return jsx(
    'span',
    {
      style: {
        ...style,
        border: '1px solid var(--ui-stroke-secondary)',
        borderRadius: '4px',
        padding: '0 6px',
        fontSize: '0.6875rem',
        lineHeight: '1.4',
      },
    },
    severity
  )
}

function FindingCard({ record }) {
  const dimmed = record.accepted === false
  return jsxs('div', {
    className: 'flex flex-col gap-1 py-2',
    style: { opacity: dimmed ? 0.55 : 1 },
    children: [
      jsxs('div', {
        className: 'flex items-center gap-2',
        children: [
          jsx(SeverityChip, { severity: record.severity }),
          record.verifier && record.verifier !== 'default'
            ? jsx('span', {
                className: 'text-xs',
                style: { color: 'var(--ui-text-tertiary)' },
                children: record.verifier,
              })
            : null,
          record.ts
            ? jsx('span', {
                className: 'text-xs',
                style: { color: 'var(--ui-text-quaternary, var(--ui-text-tertiary))' },
                children: relativeTime(record.ts * 1000),
              })
            : null,
          record.model
            ? jsx('span', {
                className: 'truncate text-xs',
                style: { color: 'var(--ui-text-quaternary, var(--ui-text-tertiary))' },
                children: record.model,
              })
            : null,
          dimmed ? jsx('span', { className: 'text-xs', children: 'suppressed' }) : null,
        ],
      }),
      jsx('div', {
        className: 'text-sm leading-relaxed whitespace-pre-wrap',
        children: record.finding,
      }),
      record.delta_excerpt
        ? jsx('div', {
            className: 'mt-1 text-xs',
            style: {
              color: 'var(--ui-text-quaternary, var(--ui-text-tertiary))',
              borderLeft: '2px solid var(--ui-stroke-secondary)',
              paddingLeft: '8px',
              overflow: 'hidden',
              maxHeight: '3em',
            },
            children: 'watched: ' + record.delta_excerpt,
          })
        : null,
    ],
  })
}

function AskCard({ record }) {
  return jsxs('div', {
    className: 'flex flex-col gap-1 py-2',
    children: [
      jsxs('div', {
        className: 'text-sm',
        style: { color: 'var(--ui-text-secondary)' },
        children: [
          jsx('span', {
            className: 'text-xs',
            style: { color: 'var(--ui-text-tertiary)' },
            children: 'you asked the verifier · ',
          }),
          record.question,
        ],
      }),
      jsx('div', {
        className: 'text-sm whitespace-pre-wrap',
        children: record.answer,
      }),
    ],
  })
}

function ModelPicker({ value, onSelect, primaryModel, disabled }) {
  const [open, setOpen] = useState(false)
  const [provider, model] = value.includes('/') ? value.split('/', 2) : ['', value]
  const current = value
    ? { model, provider, effort: '', fast: false }
    : { model: primaryModel || '', provider: '', effort: '', fast: false }

  const controller = {
    current,
    presetFor: () => ({}),
    applyPreset: () => {},
    select: async (modelSlug, providerSlug) => {
      const next = providerSlug ? `${providerSlug}/${modelSlug}` : modelSlug
      const ok = await onSelect(next)
      return ok !== false
    },
    setOptions: () => {},
  }

  const label = value || `Primary (${primaryModel || 'none'})`

  return jsx(DropdownMenu, {
    open,
    onOpenChange: setOpen,
    children: [
      jsx(DropdownMenuTrigger, {
        asChild: true,
        children: jsx(Button, {
          type: 'button',
          variant: 'ghost',
          disabled,
          className: 'max-w-64 justify-between gap-1 px-2 text-xs',
          children: jsxs('span', {
            className: 'flex items-center gap-1',
            children: [
              jsx('span', {
                className: 'truncate',
                style: { color: 'var(--ui-text-secondary)' },
                children: label,
              }),
              jsx('span', { className: 'opacity-50', children: ' ▾' }),
            ],
          }),
        }),
      }),
      jsx(DropdownMenuContent, {
        align: 'end',
        side: 'bottom',
        sideOffset: 6,
        className: 'w-72 p-0',
        children: jsx(ModelMenuCloseContext.Provider, {
          value: () => setOpen(false),
          children: jsx(ModelCatalogMenu, { controller }),
        }),
      }),
    ],
  })
}

function VerifierPage({ ctx }) {
  const qc = useQueryClient()
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [composerNote, setComposerNote] = useState(null)

  const stateQuery = useQuery({
    queryKey: ['hermes-verifier', 'state'],
    queryFn: () => ctx.rest('/state'),
    refetchInterval: 4000,
    retry: 1,
  })

  useEffect(() => {
    if (typeof ctx.socket !== 'function') return undefined
    let dispose
    try {
      dispose = ctx.socket('/events', () => {
        qc.invalidateQueries({ queryKey: ['hermes-verifier', 'state'] })
      })
    } catch (error) {
      return undefined
    }
    return () => {
      try {
        if (dispose) dispose()
      } catch (error) {
        /* noop */
      }
    }
  }, [ctx, qc])

  const onSelectModel = useCallback(
    async (next) => {
      try {
        const res = await ctx.rest('/model', { method: 'PUT', body: { model: next } })
        if (!res || res.ok === false) throw new Error((res && res.error) || 'failed')
        haptic('tap')
        qc.invalidateQueries({ queryKey: ['hermes-verifier', 'state'] })
        return true
      } catch (error) {
        host.notifyError(error, 'Could not set the verifier model')
        return false
      }
    },
    [ctx, qc]
  )

  const onToggle = useCallback(
    async (next) => {
      try {
        const res = await ctx.rest(next ? '/resume' : '/pause', {
          method: 'POST',
          body: {},
        })
        if (!res || res.ok === false) throw new Error((res && res.error) || 'failed')
        haptic('tap')
        qc.invalidateQueries({ queryKey: ['hermes-verifier', 'state'] })
      } catch (error) {
        host.notifyError(error, 'Could not toggle the verifier')
      }
    },
    [ctx, qc]
  )

  const onAsk = useCallback(async () => {
    const text = question.trim()
    if (!text || asking) return
    setAsking(true)
    setComposerNote(null)
    try {
      const res = await ctx.rest('/ask', { method: 'POST', body: { question: text } })
      if (!res || res.ok === false) throw new Error((res && res.error) || 'failed')
      haptic('tap')
      setQuestion('')
      setComposerNote(res.answer || 'Answer recorded.')
      qc.invalidateQueries({ queryKey: ['hermes-verifier', 'state'] })
    } catch (error) {
      host.notifyError(error, 'The verifier could not answer')
      setComposerNote('Ask failed — is the verifier backend loaded? (gateway restart required after install)')
    } finally {
      setAsking(false)
    }
  }, [question, asking, ctx, qc])

  const data = stateQuery.data
  const loading = stateQuery.isPending && !data
  const degraded = !loading && (!data || data.backend === 'degraded')

  const recent = data && data.recent_session
  const sessions = (data && data.sessions) || {}
  const recentState = recent ? sessions[recent] : null
  const enabled = recentState ? recentState.active : Boolean(data && data.settings && data.settings.enabled)
  const reviewing = Boolean(recentState && recentState.reviewing)
  const lastError = recentState && recentState.last_error

  let statusLabel = 'idle'
  let statusStyle = { color: 'var(--ui-text-tertiary)' }
  if (reviewing) {
    statusLabel = 'reviewing'
    statusStyle = { color: 'var(--ui-accent)' }
  } else if (degraded) {
    statusLabel = 'backend not loaded'
    statusStyle = { color: 'var(--ui-text-secondary)' }
  } else if (!enabled) {
    statusLabel = 'paused'
  } else if (lastError) {
    statusLabel = 'error'
    statusStyle = { color: 'var(--ui-text-secondary)' }
  } else if (recentState) {
    statusLabel = 'active'
  }

  const records = (data && data.transcript) || []
  const entries = [...records].reverse()

  return jsxs('div', {
    className: 'flex h-full flex-col',
    children: [
      // ── Header: Verifier Mode ────────────────────────────────────────────
      jsxs('div', {
        className: 'flex flex-col gap-2 px-4 pt-3 pb-2',
        style: { borderBottom: '1px solid var(--ui-stroke-secondary)' },
        children: [
          jsxs('div', {
            className: 'flex items-center justify-between gap-2',
            children: [
              jsxs('div', {
                className: 'flex items-center gap-2',
                children: [
                  jsx('span', { className: 'text-sm font-semibold', children: 'Verifier Mode' }),
                  jsx('span', {
                    className: 'text-xs',
                    style: { ...statusStyle, fontWeight: '600' },
                    children: statusLabel,
                  }),
                  reviewing ? jsx(GlyphSpinner, { className: 'size-3 opacity-70' }) : null,
                ],
              }),
              jsx('div', {
                className: 'flex items-center gap-2',
                children: jsx(Switch, {
                  checked: Boolean(enabled),
                  onCheckedChange: onToggle,
                }),
              }),
            ],
          }),
          // ── Model selector, underneath the title ─────────────────────────
          jsxs('div', {
            className: 'flex items-center gap-2',
            children: [
              jsx('span', {
                className: 'text-xs',
                style: { color: 'var(--ui-text-tertiary)' },
                children: 'Verifier model',
              }),
              jsx(ModelPicker, {
                value: (data && data.settings && data.settings.model) || '',
                onSelect: onSelectModel,
                primaryModel: (data && data.primary && data.primary.model) || '',
                disabled: asking || Boolean(degraded),
              }),
              lastError
                ? jsx('span', {
                    className: 'truncate text-xs',
                    style: { color: 'var(--ui-text-secondary)' },
                    children: lastError.slice(0, 80),
                  })
                : null,
            ],
          }),
        ],
      }),

      // ── Transcript body ──────────────────────────────────────────────────
      jsx(ScrollArea, {
        className: 'min-h-0 flex-1',
        children: jsx('div', {
          className: 'flex flex-col gap-1 px-4 py-2',
          children: loading
            ? jsx('div', { className: 'flex items-center gap-2 py-4 text-sm', children: 'loading verifier state…' })
            : degraded
              ? jsx('div', {
                  className: 'py-4 text-sm leading-relaxed',
                  style: { color: 'var(--ui-text-secondary)' },
                  children:
                    'The verifier backend is not loaded in this process. After installing or updating the plugin, restart the gateway once — then this screen shows reviewed deltas and findings live. Findings keep logging to disk in the meantime.',
                })
              : entries.length === 0
                ? jsx('div', {
                    className: 'py-6 text-sm',
                    style: { color: 'var(--ui-text-tertiary)' },
                    children: 'No reviews yet. Run a turn in a session with the verifier enabled and findings appear here.',
                  })
                : jsx(Fragment, {
                    children: entries.map((record, index) =>
                      jsx(Fragment, {
                        key: `${record.ts}-${index}`,
                        children: [
                          record.kind === 'ask'
                            ? jsx(AskCard, { record })
                            : jsx(FindingCard, { record }),
                          index < entries.length - 1
                            ? jsx(Separator, {})
                            : null,
                        ],
                      })
                    ),
                  }),
        }),
      }),

      // ── Composer ─────────────────────────────────────────────────────────
      jsxs('div', {
        className: 'flex flex-col gap-2 px-4 py-3',
        style: { borderTop: '1px solid var(--ui-stroke-secondary)' },
        children: [
          composerNote
            ? jsx('div', {
                className: 'max-h-24 overflow-y-auto text-xs whitespace-pre-wrap',
                style: { color: 'var(--ui-text-secondary)' },
                children: composerNote,
              })
            : null,
          jsxs('div', {
            className: 'flex items-end gap-2',
            children: [
              jsx(Textarea, {
                value: question,
                onChange: (event) => setQuestion(event.target.value),
                placeholder: 'Ask the verifier a question about the session…',
                className: 'min-h-9 flex-1 resize-none text-sm',
                rows: 2,
              }),
              jsx(Button, {
                type: 'button',
                disabled: asking || !question.trim(),
                onClick: onAsk,
                children: asking ? 'asking…' : 'Send',
              }),
            ],
          }),
        ],
      }),
    ],
  })
}

export default {
  id: 'hermes-verifier',
  name: 'Hermes Verifier',
  defaultEnabled: false,
  register(ctx) {
    ctx.register({
      id: 'page',
      area: ROUTES_AREA,
      data: { path: '/verifier' },
      render: () => jsx(VerifierPage, { ctx }),
    })

    ctx.register({
      id: 'nav',
      area: SIDEBAR_NAV_AREA,
      data: { path: '/verifier', label: 'Verifier', codicon: 'shield' },
    })

    ctx.register({
      id: 'palette',
      area: PALETTE_AREA,
      data: {
        id: 'hermes-verifier.open',
        label: 'Open Verifier Mode',
        keywords: ['verifier', 'verify', 'review'],
        run: () => host.navigate('/verifier'),
      },
    })
  },
}
