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

async function runTask(
  sdk: OpencodeClient,
  projectPath: string,
  prompt: string,
  agent?: string,
  taskId?: string
): Promise<string | undefined> {
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
      id: taskId || "unknown",
      payload: {
        code: "SESSION_CREATION_FAILED",
        message: "Failed to create session",
      },
    })
    return undefined
  }

  // Send session.started event
  sendMessage({
    type: "event",
    id: taskId || sessionId,
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

  // Event loop
  const eventLoop = async () => {
    for await (const event of events.stream) {
      if (event.type === "message.part.updated") {
        const part = event.properties.part
        if (part.sessionID !== sessionId) continue

        if (part.type === "step-start") {
          sendMessage({
            type: "event",
            id: taskId || sessionId,
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
            id: taskId || sessionId,
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
            id: taskId || sessionId,
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
          sendMessage({
            type: "event",
            id: taskId || sessionId,
            payload: {
              event: {
                type: "tool.completed",
                tool: part.tool,
                output: (part.state as { output?: Record<string, unknown> }).output,
                title: (part.state as { title?: string }).title,
              },
            },
          })
        }

        if (part.type === "text" && part.time?.end) {
          sendMessage({
            type: "event",
            id: taskId || sessionId,
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
            id: taskId || sessionId,
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
          id: taskId || sessionId,
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

        sendMessage({
          type: "error",
          id: taskId || sessionId,
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
  const model = agent ? undefined : undefined // Will use default
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

  return sessionId
}

export const StdioCommand = cmd({
  command: "stdio",
  describe: "Run opencode in stdio mode for integration with external tools",
  builder: (yargs: Argv) => {
    return yargs.option("project", {
      type: "string",
      describe: "project path to work in",
    })
  },
  handler: async (args) => {
    const projectPath = args.project || process.cwd()

    // Send ready message
    sendMessage({
      type: "ready",
      id: "init",
      payload: {
        version: "1.0.0",
        projectPath,
      },
    })

    // Read messages from stdin
    const stdin = Bun.stdin()
    const decoder = new TextDecoder()

    for await (const chunk of stdin) {
      const line = decoder.decode(chunk).trim()
      if (!line) continue

      try {
        const message: JanToOpenCode = JSON.parse(line)

        if (message.type === "task") {
          const payload = message.payload as TaskPayload
          await bootstrap(payload.projectPath || projectPath, async () => {
            const fetchFn = (async (input: RequestInfo | URL, init?: RequestInit) => {
              const request = new Request(input, init)
              return Server.App().fetch(request)
            }) as typeof globalThis.fetch
            const sdk = createOpencodeClient({ baseUrl: "http://opencode.internal", fetch: fetchFn })
            await runTask(sdk, payload.projectPath || projectPath, payload.prompt, payload.agent, message.id)
          })
        }

        // Handle other message types (permission_response, cancel, input) if needed
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
  },
})