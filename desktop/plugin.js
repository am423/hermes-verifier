/**
 * hermes-verifier — desktop half (unified package: <plugin dir>/desktop/plugin.js)
 *
 * Verifier Mode rides on a REAL session:
 *
 *  - Sidebar entry "Verifier" (visible once the plugin is enabled in
 *    Settings → Plugins) routes to /verifier, which creates/opens the
 *    designated verifier session and hands it to the native chat surface —
 *    a regular session screen, but with the verifier attached.
 *  - While that session is focused, the titlebar shows "Verifier Mode" and
 *    the Verifier model dropdown (the composer's own model catalog) plus a
 *    live status chip.
 *  - A right-side Findings pane shows the verifier's activity transcript.
 *
 * The backend (dashboard/plugin_api.py) mounts at gateway startup; until
 * then the route degrades to an explanatory note instead of a broken screen.
 */

import {
  host,
  haptic,
  useValue,
  useQuery,
  useQueryClient,
  ModelCatalogMenu,
  ModelMenuCloseContext,
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  Button,
  GlyphSpinner,
  ScrollArea,
  Separator,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  PALETTE_AREA,
  PANES_AREA,
  TITLEBAR_AREAS,
  relativeTime,
} from '@hermes/plugin-sdk'
import { jsx, jsxs, Fragment } from 'react/jsx-runtime'
import { useState, useEffect, useCallback, useRef } from 'react'

const SEVERITY_STYLE = {
  note: { color: 'var(--ui-text-tertiary)', fontWeight: '500' },
  caution: { color: 'var(--ui-text-secondary)', fontWeight: '600' },
  blocker: { color: 'var(--ui-accent)', fontWeight: '700' },
}

function SeverityChip({ severity }) {
  if (!severity) return null
  const style = SEVERITY_STYLE[severity] || SEVERITY_STYLE.note
  // `severity` belongs in children, not as the third argument: in this file the
  // third argument of jsx()/jsxs() is the React KEY (see the sibling
  // components, which all pass text via `children:`). Passing it positionally
  // gave the span a key and no content, so every severity chip rendered empty.
  return jsx('span', {
    style: {
      ...style,
      border: '1px solid var(--ui-stroke-secondary)',
      borderRadius: '4px',
      padding: '0 6px',
      fontSize: '0.6875rem',
      lineHeight: '1.4',
    },
    children: severity,
  })
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
          dimmed ? jsx('span', { className: 'text-xs', children: 'suppressed' }) : null,
        ],
      }),
      jsx('div', {
        className: 'text-sm leading-relaxed whitespace-pre-wrap',
        children: record.finding,
      }),
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
            children: 'asked · ',
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

/** Titlebar overlay: "Verifier Mode" + model dropdown while the designated
 *  verifier session is focused. Returns null on every other surface. */
function VerifierBar({ ctx }) {
  const focusedStoredId = useValue(host.state.focusedStoredSessionId)
  const qc = useQueryClient()

  const vsessionQuery = useQuery({
    queryKey: ['hermes-verifier', 'vsession'],
    queryFn: () => ctx.rest('/verifier-session'),
    refetchInterval: 30000,
    retry: 1,
  })
  const stateQuery = useQuery({
    queryKey: ['hermes-verifier', 'state'],
    queryFn: () => ctx.rest('/state'),
    refetchInterval: 5000,
    retry: 1,
  })

  // ALL hooks run before any conditional return. The early return below used to
  // sit ABOVE this useCallback, so the hook count changed the moment Verifier
  // Mode opened and React threw "Rendered more hooks than during the previous
  // render".
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

  const sid = vsessionQuery.data && vsessionQuery.data.session_id
  if (!sid || focusedStoredId !== sid) return null

  const data = stateQuery.data
  const sessionState = data && data.sessions && data.sessions[sid]
  const reviewing = Boolean(sessionState && sessionState.reviewing)
  const lastError = sessionState && sessionState.last_error
  const enabled = sessionState ? sessionState.active : Boolean(data && data.settings && data.settings.enabled)

  let chipText = 'active'
  let chipStyle = { color: 'var(--ui-text-tertiary)' }
  if (reviewing) {
    chipText = 'verifying'
    chipStyle = { color: 'var(--ui-accent)' }
  } else if (lastError) {
    chipText = 'error'
    chipStyle = { color: 'var(--ui-text-secondary)' }
  } else if (!enabled) {
    chipText = 'paused'
  }

  return jsxs('div', {
    className: 'flex items-center gap-2',
    children: [
      jsx('span', {
        className: 'text-sm font-semibold',
        children: 'Verifier Mode',
      }),
      jsx('span', {
        className: 'text-xs',
        style: { color: 'var(--ui-text-tertiary)' },
        children: 'model',
      }),
      jsx(ModelPicker, {
        value: (data && data.settings && data.settings.model) || '',
        onSelect: onSelectModel,
        primaryModel: (data && data.primary && data.primary.model) || '',
        disabled: false,
      }),
      reviewing ? jsx(GlyphSpinner, { className: 'size-3 opacity-70' }) : null,
      jsx('span', {
        className: 'text-xs',
        style: { ...chipStyle, fontWeight: '600' },
        children: chipText,
      }),
    ],
  })
}

/** Findings pane: live transcript of verifier activity. */
function FindingsPane({ ctx }) {
  const qc = useQueryClient()
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

  const data = stateQuery.data
  const records = (data && data.transcript) || []
  const entries = [...records].reverse()

  return jsx(ScrollArea, {
    className: 'h-full',
    children: jsx('div', {
      className: 'flex flex-col gap-1 px-3 py-2',
      children: entries.length === 0
        ? jsx('div', {
            className: 'py-4 text-xs',
            style: { color: 'var(--ui-text-tertiary)' },
            children: 'No findings yet. Run a turn in the Verifier session and findings land here.',
          })
        : jsx(Fragment, {
            children: entries.map((record, index) =>
              jsx(Fragment, {
                key: `${record.ts}-${index}`,
                children: [
                  record.kind === 'ask'
                    ? jsx(AskCard, { record })
                    : jsx(FindingCard, { record }),
                  index < entries.length - 1 ? jsx(Separator, {}) : null,
                ],
              })
            ),
          }),
    }),
  })
}

/** /verifier route: opens the designated verifier session in the native chat
 *  surface. This component renders only briefly; the app swaps to the session. */
function VerifierOpenPage({ ctx }) {
  const [status, setStatus] = useState('Opening Verifier session…')
  const [error, setError] = useState(null)
  const done = useRef(false)

  useEffect(() => {
    if (done.current) return
    done.current = true
    ;(async () => {
      let sid
      try {
        const res = await ctx.rest('/verifier-session')
        sid = res && res.session_id
      } catch (e) {
        /* backend may be down; fall through to create path */
      }
      if (!sid) {
        try {
          const created = await host.request('session.create', { title: 'Verifier' })
          sid = created && (created.stored_session_id || created.session_id)
        } catch (e) {
          setError('Could not create the verifier session: ' + String((e && e.message) || e))
          return
        }
        if (sid) {
          try {
            await ctx.rest('/verifier-session', { method: 'POST', body: { session_id: sid } })
          } catch (e) {
            /* designation failed; the session still opens without verifier */
          }
        }
      }
      if (!sid) {
        setError('Could not resolve a verifier session.')
        return
      }
      setStatus('Opening…')
      try {
        host.openSession(sid, { intent: 'in-place' })
      } catch (e) {
        setError('Could not open the verifier session: ' + String((e && e.message) || e))
      }
    })()
  }, [ctx])

  if (error) {
    return jsx('div', {
      className: 'flex h-full flex-col items-center justify-center gap-2 p-6 text-center',
      children: [
        jsx('div', { className: 'text-sm font-medium', children: 'Verifier Mode' }),
        jsx('div', {
          className: 'max-w-md text-sm leading-relaxed',
          style: { color: 'var(--ui-text-secondary)' },
          children: error,
        }),
        jsx('div', {
          className: 'text-xs',
          style: { color: 'var(--ui-text-tertiary)' },
          children: 'The verifier backend loads at gateway startup — restart the gateway if this persists.',
        }),
      ],
    })
  }
  return jsx('div', {
    className: 'flex h-full items-center justify-center gap-2 text-sm',
    children: [jsx(GlyphSpinner, { className: 'size-4' }), status],
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
      render: () => jsx(VerifierOpenPage, { ctx }),
    })

    ctx.register({
      id: 'nav',
      area: SIDEBAR_NAV_AREA,
      order: 60,
      data: { path: '/verifier', label: 'Verifier', codicon: 'shield' },
    })

    ctx.register({
      id: 'titlebar',
      area: TITLEBAR_AREAS.center,
      order: 300,
      render: () => jsx(VerifierBar, { ctx }),
    })

    ctx.register({
      id: 'findings-pane',
      area: PANES_AREA,
      title: 'Findings',
      data: { placement: 'right', width: '300px' },
      render: () => jsx(FindingsPane, { ctx }),
    })

    ctx.register({
      id: 'palette-open',
      area: PALETTE_AREA,
      data: {
        id: 'hermes-verifier.open',
        label: 'Verifier: open Verifier Mode session',
        keywords: ['verifier', 'verify', 'review'],
        run: () => host.navigate('/verifier'),
      },
    })
  },
}
