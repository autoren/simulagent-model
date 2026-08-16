#!/usr/bin/env python3
"""Render an offline V58 pilot-writing form after a separate release lock exists."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def render_offline_form(
    packet: dict[str, Any], packet_sha256: str, protocol: dict[str, Any]
) -> str:
    """Return a self-contained HTML form; this performs no file or network I/O."""
    packet_json = json.dumps(packet, sort_keys=True).replace("</", "<\\/")
    attestation_json = json.dumps(
        protocol["submissionSchema"]["attestation"], sort_keys=True
    ).replace("</", "<\\/")
    title = html.escape(packet["anonymous_writer_id"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>V58 pilot packet {title}</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; color: #18212b; }}
    .prompt {{ border: 1px solid #bac4cf; border-radius: 10px; padding: 1rem; margin: 1.25rem 0; }}
    .meta {{ color: #4a5968; font-size: .92rem; }}
    pre {{ white-space: pre-wrap; background: #f4f6f8; padding: .75rem; border-radius: 6px; }}
    textarea {{ width: 100%; min-height: 5rem; box-sizing: border-box; }}
    .error {{ color: #a11; font-weight: 600; }}
    button {{ font: inherit; padding: .65rem 1rem; }}
  </style>
</head>
<body>
  <h1>V58 pilot writing packet</h1>
  <p class="meta">Writer slot: <code>{title}</code><br>Packet SHA-256: <code>{packet_sha256}</code></p>
  <p>Write each utterance yourself. Do not use a language model, paraphraser, or other generative writing tool. Do not include personal information.</p>
  <div id="prompts"></div>
  <fieldset>
    <legend>Required attestation for every submission</legend>
    <label><input id="human" type="checkbox"> I wrote every response myself without generative assistance.</label><br>
    <label><input id="rights" type="checkbox"> I have the right to contribute these responses.</label><br>
    <label><input id="consent" type="checkbox"> I consent to research use under CC-BY-4.0.</label>
  </fieldset>
  <p id="error" class="error" role="alert"></p>
  <button id="download" type="button">Validate and download JSONL</button>
  <script>
  "use strict";
  const packet = {packet_json};
  const attestation = {attestation_json};
  const root = document.getElementById("prompts");
  function block(label, value) {{
    const wrap = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = label + ": ";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(value, null, 2);
    wrap.append(strong, pre);
    return wrap;
  }}
  packet.prompts.forEach((prompt, index) => {{
    const section = document.createElement("section");
    section.className = "prompt";
    const heading = document.createElement("h2");
    heading.textContent = `Prompt ${{index + 1}} of ${{packet.prompts.length}} — ${{prompt.construction_family}}`;
    section.append(heading);
    section.append(block("Entities", prompt.entity_legend));
    section.append(block("Known ontology", prompt.known_ontology_glossary));
    section.append(block("Intended semantics", prompt.intended_semantics));
    section.append(block("Instructions", prompt.writing_instructions));
    const area = document.createElement("textarea");
    area.id = "answer_" + prompt.prompt_id;
    area.maxLength = 500;
    area.required = true;
    area.setAttribute("aria-label", "Response for " + prompt.prompt_id);
    section.append(area);
    root.append(section);
  }});
  document.getElementById("download").addEventListener("click", () => {{
    const error = document.getElementById("error");
    error.textContent = "";
    if (!["human", "rights", "consent"].every(id => document.getElementById(id).checked)) {{
      error.textContent = "All attestations are required.";
      return;
    }}
    const rows = [];
    for (const prompt of packet.prompts) {{
      const text = document.getElementById("answer_" + prompt.prompt_id).value.trim();
      if (text.length < 4 || text.length > 500 || /[\\r\\n]/.test(text)) {{
        error.textContent = "Every response must be one line with 4–500 characters.";
        return;
      }}
      rows.push({{
        submission_id: "v58sub_" + crypto.randomUUID().replaceAll("-", ""),
        packet_id: prompt.packet_id,
        prompt_id: prompt.prompt_id,
        anonymous_writer_id: prompt.anonymous_writer_id,
        collection_round: prompt.collection_round,
        submitted_text: text,
        timestamp: new Date().toISOString(),
        consent_and_license_attestation: attestation
      }});
    }}
    const payload = rows.map(row => JSON.stringify(row)).join("\\n") + "\\n";
    const blob = new Blob([payload], {{type: "application/x-ndjson"}});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = packet.anonymous_writer_id + "-submissions.jsonl";
    link.click();
    URL.revokeObjectURL(link.href);
  }});
  </script>
</body>
</html>
"""


def validate_release_lock(
    release: dict[str, Any], release_path: Path, packet_seal: dict[str, Any]
) -> None:
    if (
        release.get("experiment") != "v58_pilot_release_lock"
        or not release.get("authorization", {}).get("release_pilot_packets")
        or not release.get("authorization", {}).get("collect_pilot_language")
        or release.get("authorization", {}).get("release_evaluation_packets")
        or release.get("authorization", {}).get("collect_evaluation_language")
        or release.get("authorization", {}).get("model_generated_writing_assistance")
        or file_sha256(PROJECT_ROOT / release["packet_seal"])
        != release["packet_seal_sha256"]
        or release["packet_seal_sha256"]
        != file_sha256(PROJECT_ROOT / "configs/v58-author-packet-seal.json")
        or release.get("pilot_packet_artifacts")
        != [
            row for row in packet_seal["artifacts"]
            if row["anonymous_writer_id"].startswith("pilot_writer_slot_")
        ]
    ):
        raise RuntimeError(
            f"V58 pilot release lock is invalid or insufficient: {release_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-lock", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    release_path = (PROJECT_ROOT / args.release_lock).resolve()
    packet_path = (PROJECT_ROOT / args.packet).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    release = json.loads(release_path.read_text())
    seal = json.loads(
        (PROJECT_ROOT / "configs/v58-author-packet-seal.json").read_text()
    )
    validate_release_lock(release, release_path, seal)
    artifact = next(
        (
            row for row in release["pilot_packet_artifacts"]
            if (PROJECT_ROOT / row["path"]).resolve() == packet_path
        ),
        None,
    )
    if artifact is None or file_sha256(packet_path) != artifact["sha256"]:
        raise RuntimeError("packet is not an intact release-authorized pilot packet")
    packet = json.loads(packet_path.read_text())
    protocol = json.loads((PROJECT_ROOT / seal["protocol"]).read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_offline_form(packet, artifact["sha256"], protocol))
    print(json.dumps({
        "output": str(output.relative_to(PROJECT_ROOT)),
        "output_sha256": file_sha256(output),
        "packet": artifact["path"],
        "packet_sha256": artifact["sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
