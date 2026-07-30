# Command At-User Compatibility Design

## Goal

Allow a real QQ `At` message component to be used wherever a QQ number is
accepted by the `/伪造转发` and `/伪造消息` command inputs. The component's
`qq` value is treated exactly like the user had typed that numeric QQ ID.

## Trigger Boundary

The conversion runs only inside the existing `/伪造转发` and `/伪造消息`
command handlers after AstrBot has matched the command. It does not add a
global event listener and does not affect ordinary messages, `/伪造帮助`, or
the `send_qq_forward_message` LLM tool.

## Input Reconstruction

Add a command-scoped helper that rebuilds input from `event.get_messages()`:

- `Plain` components contribute their text.
- `At` components contribute their validated numeric `qq` value.
- A leading At directed at the bot itself is ignored as a command wake-up.
- Existing legacy-command `Image` handling remains unchanged.
- If no usable component input is available, retain the current
  `event.get_message_str()` fallback.

The reconstructed text continues through the existing JSON, line DSL, and
legacy parsers. This keeps all existing validation and membership checks and
also makes forms such as these work without new parser syntax:

```text
/伪造转发 @用户: 内容
/伪造转发 [{"sender_id": @用户, "content": "内容"}]
/伪造转发 @用户: 你好[at:@另一用户]
/伪造消息 @用户 内容 | @另一用户 内容
```

## Errors And Safety

`AtAll` and any At target whose `qq` is not a positive numeric ID are rejected
with the existing user-facing ID validation. Reconstructed IDs still pass the
current-group membership check before a forward message is sent.

## Tests And Documentation

Add handler-level tests for At senders in the new and legacy commands, JSON
and inline At substitution, leading bot mentions, and invalid/non-user At
targets. Keep existing legacy image tests passing. Update command help and the
README with concise At examples.
