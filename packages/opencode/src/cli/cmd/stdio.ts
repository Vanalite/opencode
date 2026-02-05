import type { Argv } from "yargs"
import { cmd } from "./cmd"
import { UI } from "../ui"
import { bootstrap } from "../bootstrap"
import { createOpencodeClient, type OpencodeClient } from "@opencode-ai/sdk/v2"
import { Server } from "../../server/server"
import { Agent } from "../../agent/agent"
import { Provider } from "../../provider/provider"
import { EOL } from "os"

// ============================================================================
// Types for stdio protocol - Messages FROM Jan TO OpenCode (stdin)
// ============================================================================

export interface TaskPayload {
  sessionId?: string
  projectPath: string
  prompt: string
  agent?: string
}

export interface PermissionResponsePayload {
  permissionId: string
  action: "allow_once" | "allow_always" | "deny"
  message?: string
}

export interface CancelPayload {
  sessionId?: string
}

export interface InputPayload {
  text: string
}

export interface JanToOpenCode {
  type: "task" | "permission_response" | "cancel" | "input"
  id: string
  payload: TaskPayload | PermissionResponsePayload | CancelPayload | InputPayload
}

// ============================================================================
// Types for stdio protocol - Messages FROM OpenCode TO Jan (stdout)
// ============================================================================

export interface ReadyPayload {
  version: string
  projectPath: string
}

export interface OpenCodeEvent {
  type: "session.started" | "step.started" | "step.completed" | "tool.started" | "tool.completed" | "text.delta" | "text.complete" | "reasoning.delta" | "file.changed"
  sessionId?: string
  step?: number
  tool?: string
  input?: Record<string, unknown>
  output?: Record<string, unknown>
  title?: string
  text?: string
  path?: string
  diff?: string
}

export interface EventPayloadWrapper {
  event: OpenCodeEvent
}

export interface PermissionRequestPayload {
  permissionId: string
  sessionId: string
  permission: string
  patterns: string[]
  metadata?: Record<string, unknown>
  description?: string
}

export interface ResultPayload {
  sessionId: string
  status: "completed" | "cancelled" | "error"
  summary?: string
  filesChanged?: string[]
  tokensUsed?: number
  error?: string
}

export interface ErrorPayload {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface OpenCodeToJan {
  type: "ready" | "event" | "permission_request" | "result" | "error"
  id: string
  payload: ReadyPayload | EventPayloadWrapper | PermissionRequestPayload | ResultPayload | ErrorPayload
}

// ============================================================================
// Stdio Command Implementation
// ============================================================================

function sendMessage(message: OpenCodeToJan): void {
  process.stdout.write(JSON.stringify(message) + EOL)
}

// Active session state for permission_response and cancel handling
let activeSession: {
  sdk: OpencodeClient
  sessionId: string
  taskId: string
} | null = null

async function runTask(
  sdk: OpencodeClient,
  projectPath: string,
  prompt: string,
  agent?: string,
  taskId?: string
): Promise<string | undefined> {
  const msgId = taskId || "unknown"
  const rules = [
    {
      permission: "question",
      action: "deny",
      pattern: "*",
    },
    {
      permission: "plan_enter",
      action: "deny",
      pattern: "*",
    },
    {
      permission: "plan_exit",
      action: "deny",
      pattern: "*",
    },
  ]

  // Create session
  const sessionResult = await sdk.session.create({ title: undefined, permission: rules })
  const sessionId = sessionResult.data?.id

  if (!sessionId) {
    sendMessage({
      type: "error",
      id: msgId,
      payload: {
        code: "SESSION_CREATION_FAILED",
        message: "Failed to create session",
      },
    })
    return undefined
  }

  // Track active session so permission_response and cancel can reach the SDK
  activeSession = { sdk, sessionId, taskId: msgId }

  // Track files changed and final text for the result message
  const filesChanged: string[] = []
  let lastCompleteText: string | undefined

  // Send session.started event
  sendMessage({
    type: "event",
    id: msgId,
    payload: {
      event: {
        type: "session.started",
        sessionId,
      },
    },
  })

  // Subscribe to events
  const events = await sdk.event.subscribe()
  let completed = false
  let errorMessage: string | undefined
  let retryCount = 0
  const MAX_RETRIES = 3

  // Event loop
  const eventLoop = async () => {
    for await (const event of events.stream) {
      // Handle session status changes (retry, idle, busy)
      if (
        event.type === "session.status" &&
        event.properties.sessionID === sessionId
      ) {
        const status = event.properties.status
        if (status.type === "retry") {
          retryCount++
          const retryMsg = `Provider connection failed (attempt ${retryCount}): ${(status as { message?: string }).message || "retrying..."}`
          sendMessage({
            type: "event",
            id: msgId,
            payload: {
              event: {
                type: "text.delta",
                text: retryMsg,
              },
            },
          })
          // After MAX_RETRIES, abort the session to prevent infinite retry loop
          if (retryCount >= MAX_RETRIES) {
            errorMessage = `Provider unreachable after ${MAX_RETRIES} attempts. Is Jan's local API server running?`
            sendMessage({
              type: "error",
              id: msgId,
              payload: {
                code: "PROVIDER_UNREACHABLE",
                message: errorMessage,
              },
            })
            // Abort the session to stop retries
            try {
              await sdk.session.abort({ sessionID: sessionId })
            } catch (_) {
              // Ignore abort errors
            }
            break
          }
        }
      }

      if (event.type === "message.part.updated") {
        const part = event.properties.part
        if (part.sessionID !== sessionId) continue

        if (part.type === "step-start") {
          sendMessage({
            type: "event",
            id: msgId,
            payload: {
              event: {
                type: "step.started",
                step: (part as { step: number }).step,
              },
            },
          })
        }

        if (part.type === "step-finish") {
          sendMessage({
            type: "event",
            id: msgId,
            payload: {
              event: {
                type: "step.completed",
                step: (part as { step: number }).step,
              },
            },
          })
        }

        if (part.type === "tool" && part.state.status === "running") {
          sendMessage({
            type: "event",
            id: msgId,
            payload: {
              event: {
                type: "tool.started",
                tool: part.tool,
                input: (part.state as { input?: Record<string, unknown> }).input,
              },
            },
          })
        }

        if (part.type === "tool" && part.state.status === "completed") {
          const toolOutput = (part.state as { output?: Record<string, unknown> }).output
          const toolTitle = (part.state as { title?: string }).title

          sendMessage({
            type: "event",
            id: msgId,
            payload: {
              event: {
                type: "tool.completed",
                tool: part.tool,
                output: toolOutput,
                title: toolTitle,
              },
            },
          })

          // Track file changes from write/edit tools
          const writeLikeTools = ["write", "edit", "file_write", "file_edit", "patch"]
          if (writeLikeTools.some(t => part.tool.toLowerCase().includes(t))) {
            const filePath = (part.state as { input?: { file_path?: string; path?: string } }).input?.file_path
              || (part.state as { input?: { file_path?: string; path?: string } }).input?.path
            if (filePath && !filesChanged.includes(filePath)) {
              filesChanged.push(filePath)
            }
          }
        }

        if (part.type === "text" && part.time?.end) {
          lastCompleteText = part.text
          sendMessage({
            type: "event",
            id: msgId,
            payload: {
              event: {
                type: "text.complete",
                text: part.text,
              },
            },
          })
        }

        if (part.type === "text" && !part.time?.end) {
          sendMessage({
            type: "event",
            id: msgId,
            payload: {
              event: {
                type: "text.delta",
                text: part.text,
              },
            },
          })
        }
      }

      if (event.type === "permission.asked") {
        const permission = event.properties
        if (permission.sessionID !== sessionId) continue

        sendMessage({
          type: "permission_request",
          id: msgId,
          payload: {
            permissionId: permission.id,
            sessionId,
            permission: permission.permission,
            patterns: permission.patterns,
            metadata: permission.metadata,
            description: permission.description,
          },
        })
      }

      if (
        event.type === "session.status" &&
        event.properties.sessionID === sessionId &&
        event.properties.status.type === "idle"
      ) {
        completed = true
        break
      }

      if (event.type === "session.error") {
        const props = event.properties
        if (props.sessionID !== sessionId || !props.error) continue

        let errorMsg = String(props.error.name)
        if ("data" in props.error && props.error.data && "message" in props.error.data) {
          errorMsg = String(props.error.data.message)
        }

        errorMessage = errorMsg
        sendMessage({
          type: "error",
          id: msgId,
          payload: {
            code: "SESSION_ERROR",
            message: errorMsg,
          },
        })
      }
    }
  }

  // Start the event loop
  const eventPromise = eventLoop()

  // Send the prompt
  const selectedAgent = await (async () => {
    if (!agent) return undefined
    const entry = await Agent.get(agent)
    if (!entry || entry.mode === "subagent") return undefined
    return agent
  })()

  await sdk.session.prompt({
    sessionID: sessionId,
    agent: selectedAgent,
    model: undefined,
    parts: [{ type: "text", text: prompt }],
  })

  // Wait for completion
  await eventPromise

  // Send result message — this is what Jan's delegate tool waits for
  sendMessage({
    type: "result",
    id: msgId,
    payload: {
      sessionId,
      status: completed ? "completed" : errorMessage ? "error" : "cancelled",
      summary: lastCompleteText,
      filesChanged: filesChanged.length > 0 ? filesChanged : undefined,
      error: errorMessage,
    },
  })

  // Clear active session
  activeSession = null

  return sessionId
}

export const StdioCommand = cmd({
  command: "stdio",
  describe: "Run opencode in stdio mode for integration with external tools",
  builder: (yargs: Argv) => {
    return yargs
      .option("project", {
        type: "string",
        describe: "project path to work in",
      })
      .option("agent", {
        type: "string",
        describe: "agent type to use (e.g. build, plan, explore)",
      })
  },
  handler: async (args) => {
    const projectPath = args.project || process.cwd()
    const defaultAgent = args.agent as string | undefined

    // Send ready message
    sendMessage({
      type: "ready",
      id: "init",
      payload: {
        version: "1.0.0",
        projectPath,
      },
    })

    // Read messages from stdin — handle line-by-line (chunks may contain partial lines)
    const stdinStream = Bun.stdin.stream()
    const reader = stdinStream.getReader()
    const decoder = new TextDecoder()
    let buffer = ""
    let running = true

    while (running) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = value
      buffer += decoder.decode(chunk)
      const lines = buffer.split("\n")
      // Keep the last (possibly incomplete) line in the buffer
      buffer = lines.pop() || ""

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue

        try {
          const message: JanToOpenCode = JSON.parse(trimmed)

          if (message.type === "task") {
            const payload = message.payload as TaskPayload
            const taskProjectPath = payload.projectPath || projectPath
            const taskAgent = payload.agent || defaultAgent
            await bootstrap(taskProjectPath, async () => {
              const fetchFn = (async (input: RequestInfo | URL, init?: RequestInit) => {
                // Inject x-opencode-directory header so Server.App middleware
                // resolves the correct project directory instead of process.cwd()
                // Note: The SDK passes a Request object (not URL + init), so we must
                // handle both cases to preserve the request body correctly.
                let request: Request
                if (input instanceof Request) {
                  const mergedHeaders = new Headers(input.headers)
                  mergedHeaders.set("x-opencode-directory", taskProjectPath)
                  request = new Request(input, { headers: mergedHeaders })
                } else {
                  const headers = new Headers(init?.headers)
                  headers.set("x-opencode-directory", taskProjectPath)
                  request = new Request(input, { ...init, headers })
                }
                return Server.App().fetch(request)
              }) as typeof globalThis.fetch
              const sdk = createOpencodeClient({ baseUrl: "http://opencode.internal", fetch: fetchFn })
              await runTask(sdk, taskProjectPath, payload.prompt, taskAgent, message.id)
            })
          }

          if (message.type === "permission_response") {
            const payload = message.payload as PermissionResponsePayload
            if (activeSession) {
              // Map Jan's action format to OpenCode's reply format
              const replyMap: Record<string, string> = {
                allow_once: "once",
                allow_always: "always",
                deny: "reject",
              }
              const reply = replyMap[payload.action] || "reject"
              await activeSession.sdk.permission.reply({
                requestID: payload.permissionId,
                reply: reply as "once" | "always" | "reject",
              })
            }
          }

          if (message.type === "cancel") {
            if (activeSession) {
              await activeSession.sdk.session.abort({
                sessionID: activeSession.sessionId,
              })
            }
          }

          if (message.type === "input") {
            // Input messages are not yet used in OpenCode's stdio mode
            // but we handle them gracefully
            const payload = message.payload as InputPayload
            if (activeSession) {
              await activeSession.sdk.session.prompt({
                sessionID: activeSession.sessionId,
                parts: [{ type: "text", text: payload.text }],
              })
            }
          }
        } catch (e) {
          sendMessage({
            type: "error",
            id: "unknown",
            payload: {
              code: "PARSE_ERROR",
              message: e instanceof Error ? e.message : String(e),
            },
          })
        }
      }
    }
  },
})