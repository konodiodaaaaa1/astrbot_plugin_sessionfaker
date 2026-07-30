from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


class ForwardMessageError(ValueError):
    """A user-facing input or policy error."""


@dataclass(frozen=True, slots=True)
class Limits:
    max_nodes: int = 20
    max_segments_per_node: int = 12
    max_total_text_length: int = 4000
    max_text_segment_length: int = 1000
    max_images: int = 8
    max_name_length: int = 64
    max_presentation_length: int = 100


@dataclass(frozen=True, slots=True)
class Segment:
    type: str
    value: str


@dataclass(frozen=True, slots=True)
class ForwardNode:
    sender_id: str
    sender_name: str | None
    content: tuple[Segment, ...]


@dataclass(frozen=True, slots=True)
class Presentation:
    prompt: str = "[聊天记录]"
    summary: str = "查看 {count} 条转发消息"
    source: str = "聊天记录"


_INLINE_TOKEN = re.compile(
    r"(?<!\\)(?:\[(at|image|face):([^\]\r\n]+)\]|"
    r"\[CQ:(image|face),([^\]\r\n]+)\])"
)
_SENDER_ID_TOKEN = r"(?:\d+|\[At:[^\]\r\n]+\])"
_TEXTUAL_AT_ID = re.compile(r"^\[At:([^\]\r\n]+)\]$")
_DSL_NODE = re.compile(
    rf"^\s*({_SENDER_ID_TOKEN})(?:\[([^\]\r\n]+)\])?\s*:\s*(.*)$"
)
_LEGACY_NODE = re.compile(rf"^\s*({_SENDER_ID_TOKEN})\s+(.+?)\s*$", re.DOTALL)


def _error(message: str) -> ForwardMessageError:
    return ForwardMessageError(message)


def _positive_id(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise _error(f"{label}必须是正整数")
    text = str(value).strip()
    if not text.isdigit() or int(text) <= 0 or len(text) > 20:
        raise _error(f"{label}必须是有效的数字 ID")
    return text


def _sender_id(value: Any, label: str) -> str:
    text = str(value).strip()
    match = _TEXTUAL_AT_ID.fullmatch(text)
    if match:
        text = match.group(1).strip()
    return _positive_id(text, label)


def _nonnegative_id(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise _error(f"{label}必须是非负整数")
    text = str(value).strip()
    if not text.isdigit() or len(text) > 20:
        raise _error(f"{label}必须是有效的非负整数")
    return text


def _name(value: Any, limits: Limits) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _error("显示名必须是字符串")
    value = value.strip()
    if not value:
        raise _error("显示名不能为空")
    if len(value) > limits.max_name_length:
        raise _error(f"显示名不能超过 {limits.max_name_length} 个字符")
    return value


def _image_source(value: Any, *, allow_local: bool = False) -> str:
    if not isinstance(value, str):
        raise _error("图片来源必须是字符串")
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return value
    if allow_local and (
        PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    ):
        return value
    raise _error("图片仅支持有效的 HTTP/HTTPS URL 或 AstrBot 临时文件绝对路径")


def _cq_unescape(value: str) -> str:
    return (
        value.replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&#44;", ",")
        .replace("&amp;", "&")
    )


def _cq_params(value: str) -> dict[str, str]:
    params = {}
    for item in value.split(","):
        key, separator, raw_value = item.partition("=")
        if not separator or not key:
            raise _error("CQ 码参数格式错误")
        params[key] = _cq_unescape(raw_value)
    return params


def _segment_from_mapping(raw: Mapping[str, Any]) -> Segment:
    segment_type = raw.get("type")
    if segment_type == "text":
        value = raw.get("text")
        if not isinstance(value, str) or not value:
            raise _error("text 消息段必须包含非空 text")
        return Segment("text", value)
    if segment_type == "at":
        return Segment("at", _positive_id(raw.get("qq"), "@ 目标"))
    if segment_type == "image":
        return Segment("image", _image_source(raw.get("url")))
    if segment_type == "face":
        face_id = _nonnegative_id(raw.get("id"), "表情 ID")
        if int(face_id) > 65535:
            raise _error("表情 ID 不能大于 65535")
        return Segment("face", face_id)
    raise _error(f"不支持的消息段类型：{segment_type!r}")


def _append_text(segments: list[Segment], text: str) -> None:
    text = text.replace(r"\[", "[")
    if not text:
        return
    if segments and segments[-1].type == "text":
        segments[-1] = Segment("text", segments[-1].value + text)
    else:
        segments.append(Segment("text", text))


def parse_inline_content(text: str) -> tuple[Segment, ...]:
    if not isinstance(text, str):
        raise _error("节点内容必须是字符串")
    segments: list[Segment] = []
    cursor = 0
    for match in _INLINE_TOKEN.finditer(text):
        _append_text(segments, text[cursor : match.start()])
        token_type, token_value, cq_type, cq_value = match.groups()
        is_cq_code = cq_type is not None
        if is_cq_code:
            token_type = cq_type
            params = _cq_params(cq_value)
            token_value = params.get("file") if token_type == "image" else params.get("id")
            if token_value is None:
                raise _error(f"CQ {token_type} 码缺少必要参数")
        if token_type == "at":
            segments.append(Segment("at", _positive_id(token_value, "@ 目标")))
        elif token_type == "image":
            segments.append(
                Segment(
                    "image",
                    _image_source(token_value, allow_local=is_cq_code),
                )
            )
        else:
            face_id = _nonnegative_id(token_value, "表情 ID")
            if int(face_id) > 65535:
                raise _error("表情 ID 不能大于 65535")
            segments.append(Segment("face", face_id))
        cursor = match.end()
    _append_text(segments, text[cursor:])
    if not segments:
        raise _error("节点内容不能为空")
    return tuple(segments)


def _content(raw: Any) -> tuple[Segment, ...]:
    if isinstance(raw, str):
        if not raw:
            raise _error("节点内容不能为空")
        return parse_inline_content(raw)
    if not isinstance(raw, list) or not raw:
        raise _error("content 必须是非空字符串或消息段数组")
    segments: list[Segment] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise _error("消息段必须是对象")
        segments.append(_segment_from_mapping(item))
    return tuple(segments)


def _node_from_mapping(raw: Mapping[str, Any], limits: Limits) -> ForwardNode:
    if "sender_id" not in raw:
        raise _error("节点缺少 sender_id")
    if "content" not in raw:
        raise _error("节点缺少 content")
    return ForwardNode(
        sender_id=_sender_id(raw["sender_id"], "sender_id"),
        sender_name=_name(raw.get("sender_name"), limits),
        content=_content(raw["content"]),
    )


def parse_json_nodes(value: str | Sequence[Mapping[str, Any]], limits: Limits | None = None) -> list[ForwardNode]:
    limits = limits or Limits()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _error(f"JSON 格式错误：第 {exc.lineno} 行第 {exc.colno} 列") from exc
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error("JSON 顶层必须是节点数组")
    nodes = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise _error("每个节点必须是对象")
        nodes.append(_node_from_mapping(raw, limits))
    return validate_nodes(nodes, limits)


def parse_dsl(value: str, limits: Limits | None = None) -> list[ForwardNode]:
    limits = limits or Limits()
    nodes: list[ForwardNode] = []
    for line_number, line in enumerate(value.splitlines(), 1):
        if not line.strip():
            continue
        match = _DSL_NODE.fullmatch(line)
        if not match:
            raise _error(f"第 {line_number} 行格式错误，应为 QQ号[显示名]: 内容")
        sender_id, sender_name, content = match.groups()
        try:
            node = ForwardNode(
                sender_id=_sender_id(sender_id, "QQ 号"),
                sender_name=_name(sender_name, limits),
                content=parse_inline_content(content),
            )
        except ForwardMessageError as exc:
            raise _error(f"第 {line_number} 行：{exc}") from exc
        nodes.append(node)
    return validate_nodes(nodes, limits)


def parse_legacy(
    value: str,
    limits: Limits | None = None,
    images_by_node: Sequence[Sequence[str]] | None = None,
) -> list[ForwardNode]:
    limits = limits or Limits()
    parts = [part.strip() for part in value.split("|") if part.strip()]
    nodes: list[ForwardNode] = []
    for index, part in enumerate(parts):
        match = _LEGACY_NODE.fullmatch(part)
        if not match:
            raise _error(
                f"第 {index + 1} 段格式错误，应为 QQ号/[At:QQ号] 内容"
            )
        sender_id, text = match.groups()
        segments = [Segment("text", text)]
        if images_by_node and index < len(images_by_node):
            segments.extend(
                Segment("image", _image_source(url))
                for url in images_by_node[index]
            )
        nodes.append(ForwardNode(_sender_id(sender_id, "QQ 号"), None, tuple(segments)))
    return validate_nodes(nodes, limits)


def parse_command_input(value: str, limits: Limits | None = None) -> list[ForwardNode]:
    stripped = value.strip()
    if not stripped:
        raise _error("未提供转发节点")
    if stripped.startswith("["):
        return parse_json_nodes(stripped, limits)
    return parse_dsl(stripped, limits)


def validate_nodes(nodes: Sequence[ForwardNode], limits: Limits | None = None) -> list[ForwardNode]:
    limits = limits or Limits()
    if not nodes:
        raise _error("至少需要一个转发节点")
    if len(nodes) > limits.max_nodes:
        raise _error(f"节点数量不能超过 {limits.max_nodes}")
    total_text = 0
    total_images = 0
    for node_index, node in enumerate(nodes, 1):
        _positive_id(node.sender_id, f"第 {node_index} 个节点的 sender_id")
        if node.sender_name is not None:
            _name(node.sender_name, limits)
        if not node.content:
            raise _error(f"第 {node_index} 个节点内容为空")
        if len(node.content) > limits.max_segments_per_node:
            raise _error(f"第 {node_index} 个节点消息段不能超过 {limits.max_segments_per_node} 个")
        for segment in node.content:
            if segment.type == "text":
                if len(segment.value) > limits.max_text_segment_length:
                    raise _error(
                        f"第 {node_index} 个节点的单段文本不能超过 {limits.max_text_segment_length} 个字符"
                    )
                total_text += len(segment.value)
            elif segment.type == "image":
                _image_source(segment.value, allow_local=True)
                total_images += 1
            elif segment.type == "at":
                _positive_id(segment.value, "@ 目标")
            elif segment.type == "face":
                face_id = _nonnegative_id(segment.value, "表情 ID")
                if int(face_id) > 65535:
                    raise _error("表情 ID 不能大于 65535")
            else:
                raise _error(f"不支持的消息段类型：{segment.type!r}")
    if total_text > limits.max_total_text_length:
        raise _error(f"总文本长度不能超过 {limits.max_total_text_length} 个字符")
    if total_images > limits.max_images:
        raise _error(f"图片数量不能超过 {limits.max_images} 张")
    return list(nodes)


def referenced_member_ids(nodes: Iterable[ForwardNode]) -> set[str]:
    member_ids = {node.sender_id for node in nodes}
    member_ids.update(
        segment.value
        for node in nodes
        for segment in node.content
        if segment.type == "at"
    )
    return member_ids


def resolve_member_names(nodes: Sequence[ForwardNode], member_names: Mapping[str, str]) -> list[ForwardNode]:
    missing = sorted(referenced_member_ids(nodes) - member_names.keys(), key=int)
    if missing:
        raise _error(f"以下 QQ 号不是当前群成员：{', '.join(missing)}")
    return [
        replace(node, sender_name=node.sender_name or member_names[node.sender_id])
        for node in nodes
    ]


def _segment_payload(segment: Segment) -> dict[str, Any]:
    if segment.type == "text":
        return {"type": "text", "data": {"text": segment.value}}
    if segment.type == "at":
        return {"type": "at", "data": {"qq": segment.value}}
    if segment.type == "image":
        return {"type": "image", "data": {"file": segment.value}}
    if segment.type == "face":
        return {"type": "face", "data": {"id": int(segment.value)}}
    raise _error(f"不支持的消息段类型：{segment.type!r}")


def _preview(node: ForwardNode, max_length: int = 80) -> str:
    chunks = []
    for segment in node.content:
        if segment.type == "text":
            chunks.append(segment.value.replace("\n", " "))
        elif segment.type == "at":
            chunks.append(f"@{segment.value}")
        elif segment.type == "image":
            chunks.append("[图片]")
        elif segment.type == "face":
            chunks.append("[表情]")
    value = "".join(chunks).strip() or "[消息]"
    return value if len(value) <= max_length else value[: max_length - 1] + "…"


def _presentation_text(value: str, label: str, limits: Limits) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{label}不能为空")
    value = value.strip()
    if len(value) > limits.max_presentation_length:
        raise _error(f"{label}不能超过 {limits.max_presentation_length} 个字符")
    return value


def build_forward_payload(
    group_id: str | int,
    nodes: Sequence[ForwardNode],
    presentation: Presentation | None = None,
    limits: Limits | None = None,
) -> dict[str, Any]:
    limits = limits or Limits()
    nodes = validate_nodes(nodes, limits)
    group_id_text = _positive_id(group_id, "群号")
    presentation = presentation or Presentation()
    prompt = _presentation_text(presentation.prompt, "prompt", limits)
    source = _presentation_text(presentation.source, "source", limits)
    try:
        summary_value = presentation.summary.format(count=len(nodes))
    except (KeyError, ValueError) as exc:
        raise _error("summary 模板仅支持 {count} 占位符") from exc
    summary = _presentation_text(summary_value, "summary", limits)
    messages = []
    for node in nodes:
        if not node.sender_name:
            raise _error(f"QQ {node.sender_id} 尚未解析显示名")
        messages.append(
            {
                "type": "node",
                "data": {
                    "user_id": node.sender_id,
                    "nickname": node.sender_name,
                    "content": [_segment_payload(segment) for segment in node.content],
                },
            }
        )
    return {
        "group_id": int(group_id_text),
        "messages": messages,
        "prompt": prompt,
        "summary": summary,
        "source": source,
        "news": [
            {"text": f"{node.sender_name}: {_preview(node)}"}
            for node in nodes[:4]
        ],
    }
