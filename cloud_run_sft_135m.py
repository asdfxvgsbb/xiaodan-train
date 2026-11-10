# -*- coding: utf-8 -*-
"""
小蛋 v9 云端 SFT 微调启动器（带每小时上传 + 自动续跑）
========================================================
能力：
  1. 从 HF 仓库拉取最新预训练模型（30 万步）
  2. 拉取 SFT 数据（qa_all.json + corpus_dialog.txt）+ 训练脚本 + 词表
  3. 跑 SFT 微调（mode='sft'，lr=1e-3，上下文 T=64，可续跑）
  4. 每小时自动上传 checkpoint（防 GitHub 6 小时超时丢进度）
  5. 未达目标步数时自动触发下一个 run（保活续跑）

用法（任意云端 Notebook / GitHub Actions）：
    import cloud_run_sft
    cloud_run_sft.run()

环境变量（可选）：
    XDAN_TOKEN       HF token（默认内置）
    XDAN_REPO        HF 仓库（默认 jinax/xiaodan-v9）
    XDAN_SFT_STEPS   SFT 目标总步数（默认 3000）
    XDAN_LR          SFT 学习率（默认 0.001）
    XDAN_BUDGET      单次时间预算秒（默认 32400 = 9小时）
    XDAN_B           SFT 批大小（默认 12）
    XDAN_UPLOAD_SEC  上传间隔秒（默认 3600 = 1小时）
    XDAN_GH_TOKEN    GitHub token（保活续跑用）
"""

import os
import sys
import time
import json
import shutil
import threading

TOKEN = os.environ.get("XDAN_TOKEN") or os.environ.get("HF_TOKEN") or ""
REPO = os.environ.get("XDAN_REPO", "jinax/xiaodan-135m")
# 基础模型仓库：SkyPile 20000 步预训练模型所在地（与 SFT 数据仓库分开）
MODEL_REPO = os.environ.get("XDAN_MODEL_REPO", "jinax/xiaodan-pro")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

if not TOKEN:
    raise SystemExit(
        "缺少 token。请先设置环境变量：\n"
        "    os.environ['XDAN_TOKEN'] = 'hf_...'")

# 基础预训练模型 + 词表（从 MODEL_REPO 拉取，SkyPile 20000 步版）
BASE_MODEL = os.environ.get("XDAN_BASE_MODEL", "xiaodan_135m_skypile.pkl")
VOCAB_FILE = os.environ.get("XDAN_VOCAB", "vocab_skypile.json")
# SFT 训练脚本 + 数据（从 REPO 拉取）
SFT_FILES = ["train_my_ai_v8.py", "corpus_dialog.txt", "qa_all.json",
             "vocab_skypile.json"]
# 最终工作文件：模型统一命名为 xiaodan_135m.pkl（SFT 会写回这个文件）
WORK_MODEL = "xiaodan_135m.pkl"


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=TOKEN, endpoint=ENDPOINT)


def prepare():
    """下载仓库文件到本地工作目录（双仓库：基础模型 + SFT 数据）"""
    work = os.environ.get("XDAN_DIR")
    if not work:
        for cand in ("/content/xd_sft", "/kaggle/working/xd_sft",
                     os.path.join(os.getcwd(), "xd_sft"), "/tmp/xd_sft"):
            try:
                os.makedirs(cand, exist_ok=True)
                work = cand
                break
            except Exception:
                continue
    os.makedirs(work, exist_ok=True)

    from huggingface_hub import snapshot_download
    reset = os.environ.get("XDAN_SFT_RESET", "0") == "1"

    # 1) 基础模型 + 词表（从 MODEL_REPO 拉，SkyPile 20000 步）
    base_files = [BASE_MODEL, VOCAB_FILE]
    have_base = all(os.path.exists(os.path.join(work, f)) for f in base_files)
    if not have_base or reset:
        print("[下载] 从 %s 拉取基础模型 %s + 词表 %s ..." %
              (MODEL_REPO, BASE_MODEL, VOCAB_FILE))
        snapshot_download(MODEL_REPO, repo_type="model", token=TOKEN,
                          local_dir=work, endpoint=ENDPOINT,
                          allow_patterns=base_files)
    else:
        print("[本地] 基础模型 + 词表已在工作目录，跳过下载")

    # 2) SFT 训练脚本 + 数据（从 REPO 拉）
    sft_missing = [f for f in SFT_FILES if not os.path.exists(os.path.join(work, f))]
    if sft_missing:
        print("[下载] 从 %s 拉取 SFT 数据/脚本（%s）..." % (REPO, ", ".join(sft_missing)))
        snapshot_download(REPO, repo_type="model", token=TOKEN,
                          local_dir=work, endpoint=ENDPOINT,
                          allow_patterns=sft_missing)
    else:
        print("[本地] SFT 数据/脚本已在工作目录，跳过下载")

    # 3) 统一工作文件名：基础模型 -> xiaodan_135m.pkl（SFT 写回该文件）
    #    注意：仅当重置或文件缺失时才覆盖，续跑（非 reset）时保留已有 SFT 进度
    base_src = os.path.join(work, BASE_MODEL)
    work_model = os.path.join(work, WORK_MODEL)
    if not os.path.exists(base_src):
        raise SystemExit("缺少基础模型: %s（检查 MODEL_REPO=%s）" % (BASE_MODEL, MODEL_REPO))
    if reset or not os.path.exists(work_model):
        print("[工作] %s -> %s（%s）" % (BASE_MODEL, WORK_MODEL, "重置" if reset else "首次"))
        shutil.copyfile(base_src, work_model)
    else:
        print("[工作] 已有 %s，保留（续跑，跳过复制）" % WORK_MODEL)

    missing = [f for f in SFT_FILES + [VOCAB_FILE]
               if not os.path.exists(os.path.join(work, f))]
    if missing:
        raise SystemExit("缺少文件: %s" % ", ".join(missing))
    return work


def sft_step(model_path):
    """读取 SFT 累计步数（_opt['sft_t']），续跑判断用"""
    try:
        import pickle
        with open(model_path, "rb") as f:
            d = pickle.load(f)
        return int(d.get("cfg", {}).get("_opt", {}).get("sft_t", 0))
    except Exception:
        return 0


def upload_model(work, tag=""):
    """Upload model to HF, retry both official and mirror endpoints."""
    import time as _time
    mp = os.path.join(work, WORK_MODEL)
    if not os.path.exists(mp):
        return False
    endpoints = ["https://huggingface.co", "https://hf-mirror.com"]
    for ep in endpoints:
        for attempt in range(3):
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=TOKEN, endpoint=ep)
                api.upload_file(
                    path_or_fileobj=mp,
                    path_in_repo=WORK_MODEL,
                    repo_id=REPO, repo_type="model",
                    commit_message="sft checkpoint %s" % tag)
                print("[upload] checkpoint %s done via %s" % (tag, ep))
                return True
            except Exception as e:
                print("[upload] %s attempt%d failed: %s" % (ep, attempt+1, str(e)[:80]))
                _time.sleep(10)
    return False

def watch_and_upload(work, stop_flag, every_sec):
    """监控线程：每 every_sec 秒上传一次（模型步数变化才传）"""
    last = -1
    while not stop_flag.is_set():
        time.sleep(max(10, every_sec))
        cur = sft_step(os.path.join(work, WORK_MODEL))
        if cur > last:
            print("\n[监控] SFT 步数 %d，触发上传" % cur)
            upload_model(work, tag=str(cur))
            last = cur


def trigger_next_run():
    """续跑：SFT 未完成且无活跃 run 时才触发下一个 GitHub Actions run（防滚雪球）"""
    gh_token = os.environ.get("XDAN_GH_TOKEN") or ""
    if not gh_token:
        print("[续跑] 未设置 XDAN_GH_TOKEN，跳过自动续跑")
        return False
    import urllib.request, json as _json
    owner, repo = "asdfxvgsbb", "xiaodan-train"
    # 唯一性检查：先看是否已有活跃 SFT run（queued/in_progress），有就不触发
    try:
        list_url = "https://api.github.com/repos/%s/%s/actions/runs?per_page=20" % (owner, repo)
        lreq = urllib.request.Request(list_url, headers={
            "Authorization": "token " + gh_token,
            "Accept": "application/vnd.github+json",
            "User-Agent": "operit"})
        with urllib.request.urlopen(lreq, timeout=20) as r:
            data = _json.loads(r.read())
        my_run = os.environ.get("GITHUB_RUN_ID", "")
        active = [run for run in data.get("workflow_runs", [])
                  if run.get("name") == "xiaodan-sft-135m"
                  and run.get("status") in ("queued", "in_progress")
                  and str(run.get("id")) != str(my_run)]
        if active:
            print("[续跑] 已有活跃 SFT run（%d 个），不重复触发" % len(active))
            return False
    except Exception as e:
        print("[续跑] 查询活跃 run 失败: %s" % str(e)[:100])

    url = "https://api.github.com/repos/%s/%s/actions/workflows/gh_sft_135m.yml/dispatches" % (owner, repo)
    body = _json.dumps({"ref": "main"}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": "token " + gh_token,
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code = r.status
        print("[续跑] 已触发续跑 run，HTTP %d" % code)
        return code in (201, 204)
    except Exception as e:
        print("[续跑] 触发失败: %s" % str(e)[:150])
        return False


def run():
    work = prepare()

    # 环境变量：指向工作目录
    os.environ["V8_VOCAB"] = os.path.join(work, VOCAB_FILE)
    os.environ["V8_MODEL"] = os.path.join(work, WORK_MODEL)
    os.environ["V8_QA"] = os.path.join(work, "qa_all.json")
    os.environ["V8_DIALOG"] = os.path.join(work, "corpus_dialog.txt")
    os.environ["V8_B"] = os.environ.get("XDAN_B", "12")
    os.environ["V8_THREADS"] = os.environ.get("XDAN_THREADS", "8")
    os.environ.pop("V8_CPUS", None)
    # SFT 中途落盘：让监控线程能读到新步数（落盘间隔 = 上传间隔/2）
    every_sec = int(os.environ.get("XDAN_UPLOAD_SEC", "3600"))
    os.environ["V8_SAVE_EVERY"] = str(max(1, every_sec // 2))

    sft_steps = os.environ.get("XDAN_SFT_STEPS", "3000")  # 目标总步数
    budget = os.environ.get("XDAN_BUDGET", "32400")
    lr = os.environ.get("XDAN_LR", "0.001")

    start_sft = sft_step(os.environ["V8_MODEL"])
    if os.environ.get("XDAN_SFT_RESET", "0") == "1":
        print("[重置] XDAN_SFT_RESET=1，SFT 从头开始（忽略已累计 %d 步）" % start_sft)
        start_sft = 0
    remain = int(sft_steps) - start_sft
    if remain <= 0:
        print("SFT 目标 %s 步已完成（当前 %d），无需续跑" % (sft_steps, start_sft))
        return

    script = os.path.join(work, "train_my_ai_v8.py")
    sys.argv = [script, "sft", str(remain), budget, lr]

    print("=" * 62)
    print("小蛋 v9 云端 SFT 微调")
    print("  目标步数 : %s" % sft_steps)
    print("  已累计   : %d" % start_sft)
    print("  本次续训 : %d" % remain)
    print("  学习率   : %s" % lr)
    print("  上传间隔 : %d 秒" % every_sec)
    print("  时间预算 : %s 秒" % budget)
    print("=" * 62)

    stop_flag = threading.Event()
    mon = threading.Thread(target=watch_and_upload,
                           args=(work, stop_flag, every_sec), daemon=True)
    mon.start()

    t0 = time.time()
    try:
        import runpy
        runpy.run_path(script, run_name="__main__")
    except KeyboardInterrupt:
        print("\n[中断] 收到中断信号，保存并上传...")
    except Exception as e:
        print("\n[异常] %s" % str(e)[:200])
    finally:
        stop_flag.set()
        end_sft = sft_step(os.environ["V8_MODEL"])
        print("\n" + "=" * 62)
        print("SFT 结束 | 累计 %d 步 | 耗时 %.0f 秒" % (end_sft, time.time() - t0))
        print("正在上传最终模型...")
        upload_model(work, tag="final %d" % end_sft)
        print("[保活] 检查是否达到 SFT 目标...")
        target = int(sft_steps)
        if end_sft >= target:
            print("[保活] 已达 SFT 目标 %d 步，停止续跑" % target)
        else:
            print("[保活] 未达目标（%d/%d），自动触发续跑" % (end_sft, target))
            trigger_next_run()
        print("=" * 62)


if __name__ == "__main__":
    run()