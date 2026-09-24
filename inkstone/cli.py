"""Command-line interface for the Intern InkStone automation suite with multi-account support."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .client import InkStoneClient
from .deploy import SUPPORTED_RECIPES


def print_header(title: str) -> None:
    print(f"\n\033[1;36m=== {title} ===\033[0m")


def print_success(msg: str) -> None:
    print(f"\033[1;32m[✓]\033[0m {msg}")


def print_error(msg: str) -> None:
    print(f"\033[1;31m[✗]\033[0m {msg}")


def print_info(label: str, val: str) -> None:
    print(f"  \033[1m{label:<22}\033[0m: {val}")


def cmd_status(client: InkStoneClient, args: argparse.Namespace) -> None:
    if getattr(args, "all", False):
        pool = client.get_pool_status()
        print_header(f"Pooled Multi-Account Quota ({pool['accounts_count']} Accounts)")
        p = pool["pooled"]
        print_info("Total Token Credits", f"{p['total_credits']:.6f} 墨点")
        print_info("Total Approx Tokens", f"~{p['total_tokens_formatted']} tokens")
        print_info("Total Compute Points", f"{p['total_compute_points']} 算力点")
        print_info("Total Nvidia A100", f"~{p['total_a100_hours']} Hours (60 pts/hr)")
        print_info("Total Ascend 910B", f"~{p['total_ascend_hours']} Hours (53 pts/hr)")

        print_header("Individual Account Breakdown")
        print(f"  \033[1m{'ACCOUNT':<10} {'USERNAME':<22} {'SSO UID':<12} {'TOKENS (墨点)':<16} {'A100 HOURS':<12} {'EXPIRES'}\033[0m")
        print("  " + "-" * 88)
        for acc_name, st in pool["accounts"].items():
            acc = st["account"]
            tok = st.get("tokens", {})
            cmp = st.get("compute", {})
            uname = acc.get("username", "Unknown")
            uid = acc.get("sso_uid", "")
            creds = f"{tok.get('available_credits', 0.0):.4f} pts"
            a100 = f"{cmp.get('a100_hours_remaining', 0)} hrs"
            days = f"{acc.get('remaining_days', 0)}d left"
            active_marker = " *" if acc_name == client.config.active_account else ""
            print(f"  {acc_name + active_marker:<10} {uname:<22} {uid:<12} {creds:<16} {a100:<12} {days}")
        return

    status = client.get_full_status()
    acc = status["account"]
    tok = status["tokens"]
    cmp = status["compute"]

    print_header(f"Intern InkStone Account Status [{acc.get('profile_name', 'default')}]")
    print_info("Profile Name", str(acc.get("profile_name", "")))
    print_info("Username", str(acc.get("username", "")))
    print_info("SSO UID", str(acc.get("sso_uid", "Unknown")))
    print_info("Session Valid", "YES" if acc.get("token_valid") else "NO / Expired")
    print_info("Session Expires At", str(acc.get("expires_at", "Unknown")))
    print_info("Days Remaining", f"{acc.get('remaining_days', 0)} days")

    print_header("TokenPlan Quota (Cloud Inference)")
    if "error" in tok:
        print_error(f"Failed to fetch token balance: {tok['error']}")
    else:
        print_info("Available Credits", f"{tok.get('available_credits', 0.0):.6f} 墨点 (Ink Points)")
        print_info("Approx. Tokens", f"~{tok.get('approx_tokens_formatted', '0')} tokens (1 pt ≈ 20M)")
        print_info("Rate Limits", f"{tok.get('rpm_limit', 50)} RPM / {tok.get('tpm_limit', 2000000):,} TPM")

        windows = tok.get("usage_windows", {})
        if "5h" in windows and windows["5h"].get("enabled"):
            w5 = windows["5h"]
            print_info("5-Hour Rolling Limit", f"{w5.get('remaining_credits', '0')}/{w5.get('limit_credits', '0')} pts (recovers {w5.get('next_recover_at')})")
        if "7d" in windows and windows["7d"].get("enabled"):
            w7 = windows["7d"]
            print_info("7-Day Rolling Limit", f"{w7.get('remaining_credits', '0')}/{w7.get('limit_credits', '0')} pts (recovers {w7.get('next_recover_at')})")

    print_header("Compute & GPU Quota (Cloud Workbench)")
    if "error" in cmp:
        print_error(f"Failed to fetch compute quota: {cmp['error']}")
    else:
        print_info("Compute Points", f"{cmp.get('remaining_points', 0)} / {cmp.get('total_points', 0)} 算力点")
        print_info("Nvidia A100 (80GB)", f"~{cmp.get('a100_hours_remaining', 0)} Hours (60 pts/hr)")
        print_info("Ascend 910B (64GB)", f"~{cmp.get('ascend_hours_remaining', 0)} Hours (53 pts/hr)")
        print_info("Notebook Instances", f"{cmp.get('notebooks_used', 0)} used / {cmp.get('notebooks_total', 10)} max")


def cmd_accounts(client: InkStoneClient, args: argparse.Namespace) -> None:
    action = args.acc_action

    if action == "list" or not action:
        accounts = client.config.accounts
        active = client.config.active_account
        print_header(f"Configured Accounts ({len(accounts)})")
        print(f"  \033[1m{'ID':<10} {'USERNAME':<24} {'SSO UID':<14} {'ACTIVE'}\033[0m")
        print("  " + "-" * 60)
        for name, acc in accounts.items():
            uname = acc.get("username", "Unknown")
            uid = acc.get("ssouid", "")
            is_active = "\033[1;32mYES\033[0m" if name == active else "No"
            print(f"  {name:<10} {uname:<24} {uid:<14} {is_active}")

    elif action == "switch":
        target = args.target_account
        if target not in client.config.accounts:
            print_error(f"Account '{target}' not found. Choose from: {list(client.config.accounts.keys())}")
            return
        client.config.active_account = target
        client.config.save()
        print_success(f"Switched default active account to '{target}'.")


def cmd_models(client: InkStoneClient, args: argparse.Namespace) -> None:
    models = client.list_models()
    print_header(f"Available Cloud Models ({len(models)})")
    print(f"  \033[1m{'ID':<28} {'TYPE':<14} {'CONTEXT':<12} {'CAPABILITIES'}\033[0m")
    print("  " + "-" * 75)
    for m in models:
        mid = m.get("id", "")
        mtype = m.get("model_type", "")
        ctx = f"{m.get('context_length', 0):,}"
        caps = ", ".join(m.get("capabilities", []))
        print(f"  {mid:<28} {mtype:<14} {ctx:<12} {caps}")


def cmd_chat(client: InkStoneClient, args: argparse.Namespace) -> None:
    model = args.model or client.config.default_model
    prompt = " ".join(args.prompt) if isinstance(args.prompt, list) else args.prompt
    if not prompt:
        print_error("Prompt is required.")
        sys.exit(1)

    effort = getattr(args, "effort", None)
    budget = getattr(args, "budget", None)

    if args.stream:
        print(f"\033[1;34m[{model}]\033[0m ", end="", flush=True)
        generator = client.chat(
            prompt,
            model=model,
            stream=True,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=effort,
            thinking_budget=budget,
        )
        for chunk in generator:
            print(chunk, end="", flush=True)
        print()
    else:
        res = client.chat(
            prompt,
            model=model,
            stream=False,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=effort,
            thinking_budget=budget,
        )
        if res.get("reasoning_content"):
            print(f"\033[2m[Thinking]: {res['reasoning_content'].strip()}\033[0m\n")
        print(f"\033[1;34m[{model}]\033[0m {res.get('content', '')}")
        usage = res.get("usage", {})
        if usage:
            r_tok = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            print(f"\n\033[2mTokens: {usage.get('total_tokens', 0)} (prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}, reasoning: {r_tok})\033[0m")


def cmd_keys(client: InkStoneClient, args: argparse.Namespace) -> None:
    action = args.key_action

    if action == "list" or not action:
        keys = client.tokenplan.list_keys()
        print_header(f"TokenPlan API Keys ({len(keys)})")
        if not keys:
            print("  No API keys found. Run 'inkstone keys create' to generate one.")
            return
        print(f"  \033[1m{'NAME':<20} {'KEY / MASK':<24} {'STATUS':<10} {'CREATED AT'}\033[0m")
        print("  " + "-" * 75)
        for k in keys:
            name = k.get("name", "unnamed")
            mask = k.get("masked_key", k.get("key", ""))
            status = k.get("status", "unknown")
            created = k.get("created_at", "")[:19]
            print(f"  {name:<20} {mask:<24} {status:<10} {created}")

    elif action == "create":
        name = args.name
        created = client.tokenplan.create_key(name=name)
        print_success("API Key successfully created!")
        print_info("Key ID", created.get("id", ""))
        print_info("Key Name", created.get("name", ""))
        print_info("Secret Key", created.get("key", ""))
        print("\n\033[33m[!] Save this key now; it will not be shown in full again.\033[0m")

    elif action == "delete":
        if not args.key_id:
            print_error("Please provide --key-id to delete.")
            return
        if client.tokenplan.delete_key(args.key_id):
            print_success(f"Key {args.key_id} deleted.")
        else:
            print_error(f"Failed to delete key {args.key_id}.")


def cmd_compute(client: InkStoneClient, args: argparse.Namespace) -> None:
    action = args.compute_action

    if action == "list" or not action:
        machines = client.compute.list_machines()
        print_header(f"Development Machines ({len(machines)})")
        if not machines:
            print("  No dev machines found. Run 'inkstone compute create --gpu a100' to create one.")
            return
        print(f"  \033[1m{'ID':<8} {'NAME':<20} {'STATUS':<12} {'FLAVOR':<18} {'TIME LEFT'}\033[0m")
        print("  " + "-" * 75)
        for m in machines:
            mid = str(m.get("id", ""))
            name = m.get("name", "")
            status = m.get("status", "")
            rc = m.get("resource_config")
            flavor = rc.get("gpu") if isinstance(rc, dict) else (m.get("config_name") or "")
            time_left = f"{m.get('remaining_duration', 0) // 60}m" if m.get("remaining_duration") else "--"
            print(f"  {mid:<8} {name:<20} {status:<12} {flavor:<18} {time_left}")

    elif action == "flavors":
        flavors = client.compute.list_flavors()
        print_header("Compute Hardware Flavors & Point Rates")
        print(f"  \033[1m{'CONFIG NAME':<20} {'GPU / CPU':<22} {'MEMORY':<10} {'RATE (PTS/HR)'}\033[0m")
        print("  " + "-" * 75)
        for f in flavors:
            cfg = f.get("config_name", "")
            gpu = f.get("gpu") or f.get("cpu", "")
            mem = f.get("video_memory") or f.get("memory", "")
            rate = f"{f.get('calc_point_per_hour', 0)} 算力点/hr"
            print(f"  {cfg:<20} {gpu:<22} {mem:<10} {rate}")

    elif action == "create":
        name = args.name or f"dev-{args.gpu}-{int(args.hours)}h"
        print(f"[*] Creating {args.gpu.upper()} dev machine '{name}' for {args.hours} hours...")
        try:
            res = client.compute.create_machine(name=name, flavor=args.gpu, hours=args.hours)
            print_success(f"Dev machine created successfully! Machine ID: {res.get('id', 'pending')}")
        except Exception as e:
            print_error(f"Failed to create machine: {e}")

    elif action == "stop":
        if not args.machine_id:
            print_error("Please specify machine ID: inkstone compute stop <id>")
            return
        if client.compute.stop_machine(args.machine_id):
            print_success(f"Machine {args.machine_id} stopped. Point billing paused.")
        else:
            print_error(f"Failed to stop machine {args.machine_id}.")

    elif action == "start":
        if not args.machine_id:
            print_error("Please specify machine ID: inkstone compute start <id>")
            return
        hours = args.hours or 2
        if client.compute.start_machine(args.machine_id, hours=hours):
            print_success(f"Machine {args.machine_id} started for {hours} hours.")
        else:
            print_error(f"Failed to start machine {args.machine_id}.")

    elif action == "delete":
        if not args.machine_id:
            print_error("Please specify machine ID: inkstone compute delete <id>")
            return
        if client.compute.delete_machine(args.machine_id):
            print_success(f"Machine {args.machine_id} deleted.")
        else:
            print_error(f"Failed to delete machine {args.machine_id}.")

    elif action == "url":
        if not args.machine_id:
            print_error("Please specify machine ID: inkstone compute url <id>")
            return
        try:
            urls = client.compute.get_ide_urls(args.machine_id)
            print_header(f"IDE Endpoints for Machine #{args.machine_id}")
            print_info("VSCode (Code-Server)", urls.get("codeserver", "N/A"))
            print_info("JupyterLab", urls.get("jupyterlab", "N/A"))
        except Exception as e:
            print_error(f"Failed to fetch IDE URLs: {e}")

    elif action == "watch":
        if not args.machine_id:
            print_error("Please specify machine ID: inkstone compute watch <id>")
            return
        import time
        print(f"[*] Monitoring machine #{args.machine_id} until running (Ctrl+C to stop)...")
        while True:
            machines = client.compute.list_machines()
            target = next((m for m in machines if str(m.get("id")) == str(args.machine_id)), None)
            if not target:
                print_error(f"Machine #{args.machine_id} not found.")
                return
            status = target.get("status")
            queued = target.get("queued_duration", 0)
            sys.stdout.write(f"\r  [{time.strftime('%X')}] Status: \033[1;33m{status}\033[0m (Queued: {queued}s)... ")
            sys.stdout.flush()
            if status == "running":
                print()
                print_success(f"Machine #{args.machine_id} is now RUNNING!")
                try:
                    urls = client.compute.get_ide_urls(args.machine_id, pod_name=target.get("pod_name"))
                    print_header("Web IDE Endpoints")
                    print_info("VSCode (Code-Server)", urls.get("codeserver", "N/A"))
                    print_info("JupyterLab", urls.get("jupyterlab", "N/A"))
                    print("\n\033[1;32m[+] Next Step:\033[0m Open the VSCode terminal and run:")
                    print("    \033[1mbash /home/alaqmar/inkstone/recipes/deploy_qwen38_uncensored.sh\033[0m\n")
                except Exception as e:
                    print_error(f"Failed to fetch URLs: {e}")
                return
            elif status in ["failed", "stopped"]:
                print()
                print_error(f"Machine entered '{status}' state. Error: {target.get('error_message')}")
                return
            time.sleep(5)


def cmd_serve(client: InkStoneClient, args: argparse.Namespace) -> None:
    from .server import start_server
    start_server(host=args.host, port=args.port, with_tunnel=not args.no_tunnel)


def cmd_deploy(client: InkStoneClient, args: argparse.Namespace) -> None:
    recipe_name = args.recipe or "qwen3.8-uncensored"
    if recipe_name not in SUPPORTED_RECIPES:
        print_error(f"Unknown recipe '{recipe_name}'. Supported: {list(SUPPORTED_RECIPES.keys())}")
        return

    script = client.deployer.generate_setup_script(recipe_name, port=args.port)
    if args.output:
        p = client.deployer.save_recipe(recipe_name, args.output, port=args.port)
        print_success(f"Deployment script written to {p}")
    else:
        print_header(f"Deployment Script for {recipe_name}")
        print(script)


def main() -> None:
    parser = argparse.ArgumentParser(description="Intern InkStone Automation Suite CLI")
    parser.add_argument("--account", "-a", help="Select active account profile (e.g. acc1, acc2)")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # status
    p_status = subparsers.add_parser("status", help="Show account balance, quota, and expiry")
    p_status.add_argument("--all", action="store_true", help="Show pooled multi-account quota summary")
    p_status.set_defaults(func=cmd_status)

    p_quota = subparsers.add_parser("quota", help="Alias for status")
    p_quota.add_argument("--all", action="store_true", help="Show pooled multi-account quota summary")
    p_quota.set_defaults(func=cmd_status)

    # accounts
    p_acc = subparsers.add_parser("accounts", help="Manage account profiles")
    acc_sub = p_acc.add_subparsers(dest="acc_action")
    acc_sub.add_parser("list", help="List configured accounts")
    a_switch = acc_sub.add_parser("switch", help="Switch default active account")
    a_switch.add_argument("target_account", help="Account name to switch to (e.g. acc1, acc2)")
    p_acc.set_defaults(func=cmd_accounts)

    # models
    p_models = subparsers.add_parser("models", help="List available cloud models")
    p_models.set_defaults(func=cmd_models)

    # chat
    p_chat = subparsers.add_parser("chat", help="Send inference prompt to cloud model")
    p_chat.add_argument("prompt", nargs="+", help="User prompt to send")
    p_chat.add_argument("--model", "-m", help="Target model (default: qwen3.8-27b)")
    p_chat.add_argument("--stream", "-s", action="store_true", help="Stream response tokens")
    p_chat.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    p_chat.add_argument("--max-tokens", type=int, default=None, help="Max tokens")
    p_chat.add_argument("--effort", "-e", choices=["low", "medium", "high"], help="Reasoning effort level (low=instant/no thinking, medium, high)")
    p_chat.add_argument("--budget", "-b", type=int, help="Thinking token budget limit (e.g. 512, 2048, 4096)")
    p_chat.set_defaults(func=cmd_chat)

    # keys
    p_keys = subparsers.add_parser("keys", help="Manage TokenPlan API keys")
    keys_sub = p_keys.add_subparsers(dest="key_action")
    keys_sub.add_parser("list", help="List API keys")
    k_create = keys_sub.add_parser("create", help="Create new API key")
    k_create.add_argument("--name", "-n", help="Key display name")
    k_del = keys_sub.add_parser("delete", help="Delete API key")
    k_del.add_argument("key_id", help="Key ID to delete")
    p_keys.set_defaults(func=cmd_keys)

    # compute
    p_comp = subparsers.add_parser("compute", help="Manage dev machines and GPU instances")
    comp_sub = p_comp.add_subparsers(dest="compute_action")
    comp_sub.add_parser("list", help="List dev machines")
    comp_sub.add_parser("flavors", help="List hardware specs and hourly rates")
    c_create = comp_sub.add_parser("create", help="Create dev machine")
    c_create.add_argument("--gpu", choices=["a100", "910b", "cpu"], default="a100", help="Hardware flavor")
    c_create.add_argument("--hours", type=int, default=2, help="Runtime duration in hours")
    c_create.add_argument("--name", "-n", help="Machine name")
    c_start = comp_sub.add_parser("start", help="Start stopped machine")
    c_start.add_argument("machine_id", help="Machine ID")
    c_start.add_argument("--hours", type=int, default=2, help="Runtime duration in hours")
    c_stop = comp_sub.add_parser("stop", help="Stop running machine")
    c_stop.add_argument("machine_id", help="Machine ID")
    c_del = comp_sub.add_parser("delete", help="Delete machine")
    c_del.add_argument("machine_id", help="Machine ID")
    c_url = comp_sub.add_parser("url", help="Get VSCode and JupyterLab URLs")
    c_url.add_argument("machine_id", help="Machine ID")
    c_watch = comp_sub.add_parser("watch", help="Monitor machine until running and display IDE URLs")
    c_watch.add_argument("machine_id", help="Machine ID")
    p_comp.set_defaults(func=cmd_compute)

    # deploy
    p_dep = subparsers.add_parser("deploy", help="Generate OSS model deployment scripts")
    p_dep.add_argument("recipe", nargs="?", choices=list(SUPPORTED_RECIPES.keys()), default="qwen3.8-uncensored")
    p_dep.add_argument("--port", type=int, default=8000, help="Local server port")
    p_dep.add_argument("--output", "-o", help="Write script to file")
    p_dep.set_defaults(func=cmd_deploy)

    # serve
    p_srv = subparsers.add_parser("serve", help="Launch OpenAI-compatible API server with Cloudflare tunnel")
    p_srv.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    p_srv.add_argument("--host", default="0.0.0.0", help="Host binding (default: 0.0.0.0)")
    p_srv.add_argument("--no-tunnel", action="store_true", help="Disable public Cloudflare tunnel")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    client = InkStoneClient(account=args.account)
    args.func(client, args)


if __name__ == "__main__":
    main()
