import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Workbook } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../../..");
const sourceDir = path.join(repoRoot, "analysis", "data_recovery", "ZYJisBoss");
const reportDir = path.join(repoRoot, "reports", "data_recovery", "outcome_recovery");
const evidenceDir = path.join(here, "evidence", "ZYJisBoss");

const activeDeployment = String.raw`E:\BaiduNetdiskDownload\使命召唤\OnSite-B-1150-Unified\five_axis\race_guidance_control\routes\five_axis_ay_v1_release\active_deployment.json`;
const specialManifest = String.raw`E:\BaiduNetdiskDownload\使命召唤\OnSite-B-1150-Unified\six_axis\UE_pytorch\release_dist_v0617\planning_cache\special_action_manifest.json`;

const outcomePath = path.join(here, "ZYJisBoss_outcome_recovery.csv");
const transferPath = path.join(here, "ZYJisBoss_files_to_transfer.csv");
const reportPath = path.join(reportDir, "ZYJisBoss_32条结果回查报告.md");
const allowedStatuses = new Set([
  "recovered_passed",
  "recovered_failed",
  "has_metrics_no_final_label",
  "aggregate_only",
  "no_result_found",
  "ambiguous_result",
]);

async function csvRows(file) {
  const text = await fs.readFile(file, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName: "data" });
  const values = workbook.worksheets.getItem("data").getUsedRange().values;
  const headers = values[0].map((value, index) => index === 0 ? String(value).replace(/^\uFEFF/, "") : String(value));
  return values.slice(1).filter((row) => row.some((value) => String(value) !== "")).map((row) =>
    Object.fromEntries(headers.map((header, index) => [header, String(row[index] ?? "")]))
  );
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function makeCsv(rows, fields) {
  return "\ufeff" + [fields, ...rows.map((row) => fields.map((field) => row[field] ?? ""))]
    .map((row) => row.map(csvEscape).join(","))
    .join("\r\n") + "\r\n";
}

async function sha256(file) {
  return createHash("sha256").update(await fs.readFile(file)).digest("hex");
}

function routeStructure(row) {
  if (row.vehicle_structure.startsWith("five_axis")) return "five_axis";
  if (row.vehicle_structure.startsWith("six_axis")) return "six_axis";
  return "unknown_or_conflict";
}

function listRows(rows) {
  return rows.map((row) => `- \`${row.sample_key}\`：${row.map}（${routeStructure(row)}）`).join("\n");
}

async function main() {
  const confirmed = await csvRows(path.join(sourceDir, "confirmed_new_unlabeled_routes.csv"));
  const unresolved = await csvRows(path.join(sourceDir, "unresolved_candidates.csv"));
  if (confirmed.length !== 32 || unresolved.length !== 9) {
    throw new Error(`expected 32 confirmed and 9 unresolved rows, got ${confirmed.length} and ${unresolved.length}`);
  }

  const active = JSON.parse(await fs.readFile(activeDeployment, "utf8"));
  const special = JSON.parse(await fs.readFile(specialManifest, "utf8"));
  const activeHash = await sha256(activeDeployment);
  const specialHash = await sha256(specialManifest);
  const activeRouteHashes = new Set(Object.values(active.routes ?? {}).map((value) => String(value).toLowerCase()));
  const specialRouteEntries = new Map(
    Object.values(special.maps ?? {}).map((entry) => [String(entry.route_file_sha256 ?? "").toLowerCase(), entry])
  );

  const sourceFields = Object.keys(confirmed[0]);
  const outcomeFields = [
    "outcome_recovery_status",
    "outcome_label",
    "outcome_source_path",
    "outcome_source_sha256",
    "outcome_match_method",
    "outcome_confidence",
    "outcome_notes",
  ];
  const all = [...confirmed, ...unresolved].map((row) => {
    const routeHash = row.route_file_sha256.toLowerCase();
    if (activeRouteHashes.has(routeHash)) {
      return {
        ...row,
        outcome_recovery_status: "aggregate_only",
        outcome_label: "",
        outcome_source_path: activeDeployment,
        outcome_source_sha256: activeHash,
        outcome_match_method: "route_file_sha256_in_deployment_manifest",
        outcome_confidence: "high",
        outcome_notes: "路线哈希明确列入部署清单，但 offline_replay 仅给出批次 9/9 汇总，不能分配逐样本通过标签。",
      };
    }
    const specialEntry = specialRouteEntries.get(routeHash);
    if (specialEntry) {
      return {
        ...row,
        outcome_recovery_status: "has_metrics_no_final_label",
        outcome_label: "",
        outcome_source_path: specialManifest,
        outcome_source_sha256: specialHash,
        outcome_match_method: "route_file_sha256_in_special_action_manifest",
        outcome_confidence: "high",
        outcome_notes: `找到路线级 global_validation 指标（continuous_boundary_clearance_m=${specialEntry.global_validation?.continuous_boundary_clearance_m ?? ""}，continuous_pcd_clearance_m=${specialEntry.global_validation?.continuous_pcd_clearance_m ?? ""}），但没有最终通过/失败字段。`,
      };
    }
    return {
      ...row,
      outcome_recovery_status: "no_result_found",
      outcome_label: "",
      outcome_source_path: "",
      outcome_source_sha256: "",
      outcome_match_method: "targeted_nearby_and_hash_search",
      outcome_confidence: "none",
      outcome_notes: "在路线邻近目录、结果目录和限定项目根的结果类文本中未找到可关联结果。",
    };
  });

  if (all.some((row) => !allowedStatuses.has(row.outcome_recovery_status))) {
    throw new Error("unexpected outcome status");
  }
  if (all.some((row) => row.outcome_label !== "")) {
    throw new Error("no final label was recovered, so outcome_label must stay blank");
  }

  const aggregateRows = all.filter((row) => row.outcome_recovery_status === "aggregate_only");
  const metricRows = all.filter((row) => row.outcome_recovery_status === "has_metrics_no_final_label");
  const noResultRows = all.filter((row) => row.outcome_recovery_status === "no_result_found");
  if (aggregateRows.length !== 12 || metricRows.length !== 1 || noResultRows.length !== 28) {
    throw new Error(`unexpected status counts: aggregate=${aggregateRows.length}, metrics=${metricRows.length}, none=${noResultRows.length}`);
  }

  await fs.mkdir(evidenceDir, { recursive: true });
  await fs.mkdir(reportDir, { recursive: true });
  const activeRepoPath = path.join(evidenceDir, "active_deployment.json");
  const specialRepoPath = path.join(evidenceDir, "six_axis_special_action_manifest.json");
  await fs.copyFile(activeDeployment, activeRepoPath);
  await fs.copyFile(specialManifest, specialRepoPath);
  if (await sha256(activeRepoPath) !== activeHash || await sha256(specialRepoPath) !== specialHash) {
    throw new Error("copied evidence hash mismatch");
  }

  await fs.writeFile(outcomePath, makeCsv(all, [...sourceFields, ...outcomeFields]), "utf8");

  const transferFields = [
    "file_role",
    "source_path_local",
    "source_sha256",
    "file_size_bytes",
    "related_sample_count",
    "related_sample_keys",
    "outcome_evidence_type",
    "required_for_training",
    "transfer_recommended",
    "repo_path",
    "transfer_notes",
  ];
  const transfers = [
    {
      file_role: "aggregate_result_report",
      source_path_local: activeDeployment,
      source_sha256: activeHash,
      file_size_bytes: String((await fs.stat(activeDeployment)).size),
      related_sample_count: String(aggregateRows.length),
      related_sample_keys: aggregateRows.map((row) => row.sample_key).join(";"),
      outcome_evidence_type: "aggregate_only",
      required_for_training: "false",
      transfer_recommended: "true",
      repo_path: path.relative(repoRoot, activeRepoPath).replaceAll(path.sep, "/"),
      transfer_notes: "用于审计路线哈希与部署批次汇总的关联；不能生成逐样本标签。",
    },
    {
      file_role: "route_metrics_report",
      source_path_local: specialManifest,
      source_sha256: specialHash,
      file_size_bytes: String((await fs.stat(specialManifest)).size),
      related_sample_count: String(metricRows.length),
      related_sample_keys: metricRows.map((row) => row.sample_key).join(";"),
      outcome_evidence_type: "has_metrics_no_final_label",
      required_for_training: "false",
      transfer_recommended: "true",
      repo_path: path.relative(repoRoot, specialRepoPath).replaceAll(path.sep, "/"),
      transfer_notes: "用于审计路线级净空指标；没有最终通过/失败字段。",
    },
  ];
  await fs.writeFile(transferPath, makeCsv(transfers, transferFields), "utf8");

  const confirmedAggregate = aggregateRows.filter((row) => row.classification === "confirmed_new_unlabeled");
  const confirmedNoResult = noResultRows.filter((row) => row.classification === "confirmed_new_unlabeled");
  const unresolvedMetric = metricRows.filter((row) => row.classification === "unresolved");
  const unresolvedNoResult = noResultRows.filter((row) => row.classification === "unresolved");
  const report = `# ZYJisBoss 32 条结果回查报告

生成日期：2026-09-19

## 结论

- 回查对象：32 条 \`confirmed_new_unlabeled\` 和 9 条 \`unresolved\`。
- 找回明确通过标签：0 条。
- 找回明确失败标签：0 条。
- 只有路线级指标：1 条，来自 9 条未确认路线。
- 只有部署聚合摘要：12 条，均来自 32 条新增无标签路线。
- 完全没有结果：28 条，其中新增无标签路线 20 条、未确认路线 8 条。
- 多结果冲突：0 条。
- 9 条未确认路线中，恢复最终结果标签 0 条；1 条补到路线级指标，但仍不能认定通过或失败，也不能解决其主数据去重状态。
- 新增真实标签训练样本：五轴 0 条，六轴 0 条。

本轮没有把部署汇总数字、路线净空指标或模型输出当作逐样本真实标签。

## 32 条新增无标签路线

### 只有聚合摘要：12 条

\`active_deployment.json\` 的 \`routes\` 字段按 \`route_file_sha256\` 明确关联以下路线，但 \`offline_replay\` 仅给出批次 \`completed=9\`、\`hard_certificate_passed=9\`，不能判断 12 条中具体哪 9 条通过。

${listRows(confirmedAggregate)}

### 没有找到结果：20 条

${listRows(confirmedNoResult)}

## 9 条未确认路线

### 找到指标但没有最终标签：1 条

${listRows(unresolvedMetric)}

该路线通过 \`route_file_sha256\` 命中六轴 \`special_action_manifest.json\` 的 \`cross_1\` 项，报告包含连续边界净空和点云净空等 \`global_validation\` 指标，但没有最终通过/失败字段。

### 没有找到结果：8 条

${listRows(unresolvedNoResult)}

## 补传文件

需要补传并已纳入本分支的结果证据共 2 份：

- \`analysis/data_recovery/outcome_recovery/evidence/ZYJisBoss/active_deployment.json\`：支撑 12 条 \`aggregate_only\` 结论。
- \`analysis/data_recovery/outcome_recovery/evidence/ZYJisBoss/six_axis_special_action_manifest.json\`：支撑 1 条 \`has_metrics_no_final_label\` 结论。

原始 NPZ 需要补传：0 份。本轮没有恢复任何真实通过/失败标签，因此没有新增训练样本需要随结果报告补传。与两份证据关联的 13 条路线仍保留在回查 CSV 中，可按原始路径和 SHA-256 复核；不重复上传无标签 NPZ。

## 回查范围与方法

未重新扫描全盘。先复用既有 \`local_file_inventory.csv\`，再限定在 41 条路线的邻近目录和三个已知项目根中：

- 检查指定的 9 类 JSON 报告文件名；
- 检查 \`certificate\`、\`validation\`、\`reports\`、\`outputs\`、\`runs\`、\`results\` 等结果目录；
- 用 41 条 \`route_file_sha256\` 对 JSON、CSV、JSONL、日志和文本做精确内容匹配；
- 发现的车辆标定验收报告和通用性能曲线没有目标路线哈希，未用于结果分类。

逐条状态、证据路径、证据哈希和匹配方法见 \`analysis/data_recovery/outcome_recovery/ZYJisBoss_outcome_recovery.csv\`；补传文件见 \`analysis/data_recovery/outcome_recovery/ZYJisBoss_files_to_transfer.csv\`。
`;
  await fs.writeFile(reportPath, report, "utf8");

  const outcomeCheck = await Workbook.fromCSV(await fs.readFile(outcomePath, "utf8"), { sheetName: "outcomes" });
  const transferCheck = await Workbook.fromCSV(await fs.readFile(transferPath, "utf8"), { sheetName: "transfers" });
  const outcomeValues = outcomeCheck.worksheets.getItem("outcomes").getUsedRange().values;
  const transferValues = transferCheck.worksheets.getItem("transfers").getUsedRange().values;
  if (outcomeValues.length !== 42 || transferValues.length !== 3) {
    throw new Error("CSV round-trip row count mismatch");
  }

  console.log(JSON.stringify({
    total: all.length,
    confirmed: confirmed.length,
    unresolved: unresolved.length,
    aggregate_only: aggregateRows.length,
    has_metrics_no_final_label: metricRows.length,
    no_result_found: noResultRows.length,
    recovered_passed: 0,
    recovered_failed: 0,
    transfer_files: transfers.length,
  }, null, 2));
}

await main();
