"""Assistente de IA OPCIONAL do ReaDroid (OpenRouter+opencode OU Codex).

Regras deste módulo:
- O núcleo do ReaDroid nunca depende dele: o app.py só o importa quando você abre o menu 13,
  e se este arquivo for apagado o menu 13 simplesmente some.
- Usa só a biblioteca padrão. A chave do OpenRouter (provedor opencode) fica FORA do projeto
  (~/.readroid), com permissão 600, para nunca ir parar num commit do GitHub.
- Dois provedores de IA, escolhidos no menu (opção [P]):
    * opencode  -> ReaDroid gerencia a chave do OpenRouter e gera o opencode.json sozinho.
    * codex     -> usa o binário `codex` já configurado globalmente no seu Termux (o ReaDroid
                   não mexe em chave/modelo do Codex, só chama o binário na pasta certa).
- Os dois são sempre iniciados dentro de uma pasta de trabalho explícita (nunca na pasta
  pessoal). Era isso que fazia a IA "abrir mas não responder" fora da pasta certa.
"""
from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HOME_DIR = Path(os.environ.get("READROID_HOME") or (Path.home() / ".readroid"))
KEY_FILE = HOME_DIR / "openrouter.key"
OC_CONFIG = HOME_DIR / "opencode.json"
SETTINGS_FILE = HOME_DIR / "ai.json"
WORKSPACE = HOME_DIR / "workspace"

OPENROUTER_API = "https://openrouter.ai/api/v1"
# IDs gratuitos vistos no catálogo do OpenRouter em out/2026. Modelos :free saem do ar com
# frequência (o nano-30b deixou de ser gratuito), por isso o ReaDroid também troca sozinho.
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
PREFERRED_MODELS = (
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)

PROVIDER_OPENCODE = "opencode"
PROVIDER_CODEX = "codex"
PROVIDERS = (PROVIDER_OPENCODE, PROVIDER_CODEX)
DEFAULT_PROVIDER = PROVIDER_OPENCODE

PROVIDER_LABELS = {
    PROVIDER_OPENCODE: "opencode (OpenRouter, gerenciado pelo ReaDroid)",
    PROVIDER_CODEX: "codex (binário global do seu Termux)",
}

# Comando do Codex em modo automático: roda a tarefa sozinho, editando arquivos sem
# perguntar de volta. Ajustável em ai.json (chave "codex_exec_args") caso sua versão do
# codex use flags diferentes — o ReaDroid nunca sobrescreve se você já tiver configurado.
DEFAULT_CODEX_EXEC_ARGS = ["exec", "--full-auto", "--skip-git-repo-check"]

_CTRL_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{20,200}$")
_BAD_MODEL_WORDS = ("embed", "rerank", "guard", "safety", "-vl", "omni", "parse")
_UNAVAIL_RE = re.compile(r"unavailable for free|slug instead|no endpoints found|model.{0,30}(not found|not available)",
                         re.IGNORECASE)


def _clean(text: Any) -> str:
    """Remove escapes ANSI/controle: saída de IA e de rede é conteúdo não confiável."""
    return _CTRL_RE.sub("", str(text))


# ------------------------------------------------------------------ chave e configuração

def valid_key_format(key: str) -> bool:
    return bool(_KEY_RE.match(key))


def save_key(key: str) -> None:
    """Grava a chave com permissão 600 (só você lê)."""
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(HOME_DIR, 0o700)
    except OSError:
        pass
    fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(key.strip() + "\n")
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass


def load_key() -> str | None:
    env = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env:
        return env
    try:
        value = KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _load_settings() -> dict[str, Any]:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_settings(patch: dict[str, Any]) -> None:
    """Lê o ai.json, aplica só as chaves de patch e regrava (nunca apaga as outras)."""
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_settings()
    data.update(patch)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model() -> str:
    model = str(_load_settings().get("model", "")).strip()
    return model or DEFAULT_MODEL


def save_model(model: str) -> None:
    _save_settings({"model": model})


def load_provider() -> str:
    provider = str(_load_settings().get("provider", "")).strip()
    return provider if provider in PROVIDERS else DEFAULT_PROVIDER


def save_provider(provider: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"Provedor desconhecido: {provider}")
    _save_settings({"provider": provider})


def load_codex_bin() -> str:
    return str(_load_settings().get("codex_bin", "") or "codex").strip() or "codex"


def load_codex_exec_args() -> list[str]:
    args = _load_settings().get("codex_exec_args")
    return [str(a) for a in args] if isinstance(args, list) and args else list(DEFAULT_CODEX_EXEC_ARGS)


def build_opencode_config(model: str) -> dict[str, Any]:
    """Config mínima e conservadora do opencode (só chaves documentadas).

    A chave NÃO vai neste arquivo: o provedor OpenRouter do opencode lê OPENROUTER_API_KEY
    do ambiente, e o ReaDroid injeta essa variável ao iniciar o processo."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"openrouter/{model}",
        "small_model": f"openrouter/{model}",
        "provider": {"openrouter": {"models": {model: {}}}},
    }


def write_opencode_config(model: str) -> Path:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    OC_CONFIG.write_text(json.dumps(build_opencode_config(model), indent=2) + "\n", encoding="utf-8")
    return OC_CONFIG


def configure(key: str, model: str | None = None) -> dict[str, Any]:
    """Configuração automática: valida o formato, guarda a chave e gera o opencode.json."""
    key = key.strip()
    if not valid_key_format(key):
        return {"ok": False, "message": "Isso não parece uma chave de API (sem espaços, 20+ caracteres)."}
    model = (model or load_model()).strip()
    save_key(key)
    save_model(model)
    path = write_opencode_config(model)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "model": model, "config": str(path),
            "message": f"Chave salva em {KEY_FILE} (permissão 600) e opencode configurado."}


def disable() -> list[str]:
    """Apaga chave, config e preferências. O restante do ReaDroid não é afetado."""
    removed = []
    for path in (KEY_FILE, OC_CONFIG, SETTINGS_FILE):
        try:
            path.unlink()
            removed.append(str(path))
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return removed


# ------------------------------------------------------------------ opencode

def find_opencode() -> str | None:
    found = shutil.which("opencode")
    if found:
        return found
    candidate = Path.home() / ".opencode" / "bin" / "opencode"
    return str(candidate) if candidate.is_file() else None


def find_codex() -> str | None:
    """O Codex não é gerenciado pelo ReaDroid: usa o binário já configurado no Termux."""
    found = shutil.which(load_codex_bin())
    if found:
        return found
    candidate = Path.home() / ".codex" / "bin" / "codex"
    return str(candidate) if candidate.is_file() else None


def codex_command(binary: str, prompt: str | None = None) -> list[str]:
    """Sem prompt: abre a TUI interativa do Codex. Com prompt: `codex exec` em modo
    automático (--full-auto) — ele edita os arquivos sozinho, sem perguntar de volta."""
    if prompt is None:
        return [binary]
    return [binary, *load_codex_exec_args(), prompt]


def find_provider_binary(provider: str) -> str | None:
    return find_opencode() if provider == PROVIDER_OPENCODE else find_codex()


def safe_workdir(path: Path) -> tuple[bool, str]:
    """Recusa pastas enormes (HOME, raiz). Abrir o opencode na pasta pessoal faz ele
    tratar o Termux inteiro como projeto, e é uma das causas de travar sem responder."""
    resolved = path.resolve()
    if resolved in (Path.home().resolve(), Path("/"), Path("/data/data/com.termux/files")):
        return False, "Escolha a pasta de um APK (output/<app>_apktool), não a pasta pessoal."
    if not resolved.is_dir():
        return False, "Pasta não encontrada."
    return True, ""


def build_env(key: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = key
    env["OPENCODE_CONFIG"] = str(OC_CONFIG)
    return env


def opencode_command(binary: str, model: str, prompt: str | None = None) -> list[str]:
    """Sem prompt: abre a interface interativa. Com prompt: `opencode run` (uma pergunta)."""
    model_ref = f"openrouter/{model}"
    if prompt is None:
        return [binary, "-m", model_ref]
    return [binary, "run", "-m", model_ref, prompt]


def _ready(provider: str) -> tuple[bool, str, str | None]:
    """Checa se o provedor está pronto para uso. Devolve (ok, mensagem_de_erro, binário)."""
    binary = find_provider_binary(provider)
    if provider == PROVIDER_OPENCODE:
        if not load_key():
            return False, "IA não configurada. Use a opção de configurar.", None
        if not binary:
            return False, "opencode não encontrado no PATH.", None
    else:
        if not binary:
            return False, ("codex não encontrado no PATH. Confirme que `codex` funciona "
                           "direto no Termux fora do ReaDroid."), None
    return True, "", binary


def _command_for(provider: str, binary: str, prompt: str | None) -> list[str]:
    if provider == PROVIDER_OPENCODE:
        return opencode_command(binary, load_model(), prompt)
    return codex_command(binary, prompt)


def _env_for(provider: str) -> dict[str, str]:
    if provider == PROVIDER_OPENCODE:
        key = load_key()
        return build_env(key) if key else os.environ.copy()
    # Codex usa a configuração global já existente no Termux: herda o ambiente como está.
    return os.environ.copy()


def open_interactive(workdir: Path) -> dict[str, Any]:
    provider = load_provider()
    ready, why, binary = _ready(provider)
    if not ready:
        return {"ok": False, "message": why}
    ok, why = safe_workdir(workdir)
    if not ok:
        return {"ok": False, "message": why}
    if provider == PROVIDER_OPENCODE:
        write_opencode_config(load_model())
    label = PROVIDER_LABELS[provider]
    try:
        code = subprocess.run(_command_for(provider, binary, None), cwd=workdir, env=_env_for(provider),
                              check=False).returncode
    except OSError as exc:
        return {"ok": False, "message": f"Não foi possível iniciar o {provider}: {exc}"}
    return {"ok": True, "message": f"{label} encerrado (código {code})."}


def ask_once(workdir: Path, prompt: str, timeout: int = 300, emit=print) -> dict[str, Any]:
    """Faz uma pergunta/tarefa no provedor ativo e imprime a saída linha a linha, já
    sanitizada. No provedor codex isto roda `codex exec --full-auto`: a IA edita os
    arquivos sozinha, sem pausar para confirmar cada passo."""
    provider = load_provider()
    ready, why, binary = _ready(provider)
    if not ready:
        return {"ok": False, "message": why}
    ok, why = safe_workdir(workdir)
    if not ok:
        return {"ok": False, "message": why}
    if provider == PROVIDER_OPENCODE:
        write_opencode_config(load_model())
    import threading
    try:
        proc = subprocess.Popen(_command_for(provider, binary, prompt), cwd=workdir, env=_env_for(provider),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", stdin=subprocess.DEVNULL)
    except OSError as exc:
        return {"ok": False, "message": f"Não foi possível iniciar o {provider}: {exc}"}
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    lines = 0
    unavailable = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            text = _clean(line.rstrip("\n"))
            if provider == PROVIDER_OPENCODE and _UNAVAIL_RE.search(text):
                unavailable = True
            emit(text)
            lines += 1
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        return {"ok": False, "message": "Interrompido."}
    finally:
        timer.cancel()
    if unavailable:
        return {"ok": False, "model_unavailable": True,
                "message": "O modelo escolhido não está mais disponível no plano gratuito."}
    if proc.returncode in (-9, 137):
        return {"ok": False, "message": f"Sem resposta em {timeout}s; processo encerrado."}
    return {"ok": proc.returncode == 0 and lines > 0,
            "message": "Concluído." if proc.returncode == 0 and lines else f"O {provider} não devolveu resposta."}


# ------------------------------------------------------------------ OpenRouter direto (diagnóstico)

def _request(url: str, key: str | None = None, payload: dict | None = None, timeout: int = 45) -> Any:
    headers = {"Content-Type": "application/json", "User-Agent": "ReaDroid"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 (URL fixa https)
        return json.loads(response.read().decode("utf-8", errors="replace"))


def explain_http_error(code: int) -> str:
    return {
        401: "Chave recusada pelo OpenRouter (inválida, apagada ou com erro de digitação).",
        402: "Conta sem crédito para este modelo. Escolha um modelo :free.",
        404: "Modelo indisponível no OpenRouter. Use 'trocar modelo' para escolher outro :free.",
        408: "Tempo esgotado no OpenRouter. Tente de novo.",
        429: "Limite do plano gratuito atingido. Espere alguns minutos ou troque de modelo.",
    }.get(code, f"O OpenRouter respondeu com erro HTTP {code}.")


def test_api(key: str, model: str) -> dict[str, Any]:
    """Testa chave + modelo direto na API, sem opencode. Separa 'problema de chave/modelo'
    de 'problema do opencode'."""
    payload = {"model": model, "max_tokens": 64,
               "messages": [{"role": "user", "content": "Responda apenas: OK"}]}
    try:
        data = _request(f"{OPENROUTER_API}/chat/completions", key, payload)
    except urllib.error.HTTPError as exc:
        return {"ok": False, "model_unavailable": exc.code in (400, 404),
                "message": explain_http_error(exc.code)}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "message": f"Sem conexão com o OpenRouter ({_clean(exc)[:80]})."}
    except ValueError:
        return {"ok": False, "message": "Resposta inesperada do OpenRouter."}
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message") if isinstance(err, dict) else err
        return {"ok": False, "model_unavailable": bool(_UNAVAIL_RE.search(str(msg))),
                "message": _clean(msg)[:200] or "Erro devolvido pelo OpenRouter."}
    return {"ok": True, "message": "Chave e modelo funcionando."}


def pick_free_models(models: list[dict]) -> list[str]:
    """Filtra, do catálogo, modelos de chat Nemotron gratuitos."""
    found = []
    for item in models:
        mid = str(item.get("id", ""))
        low = mid.lower()
        if "nemotron" in low and mid.endswith(":free") and not any(w in low for w in _BAD_MODEL_WORDS):
            found.append(mid)
    rank = {m: i for i, m in enumerate(PREFERRED_MODELS)}
    return sorted(found, key=lambda m: (rank.get(m, len(rank)), m))


def list_free_nemotron() -> list[str]:
    try:
        data = _request(f"{OPENROUTER_API}/models", timeout=20)
        return pick_free_models(data.get("data", []))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, AttributeError):
        return []


def ensure_working_model(key: str, tried: tuple[str, ...] = (), max_tests: int = 4) -> str | None:
    """Procura um modelo Nemotron :free que responda, salva e reescreve a config.
    Só considera IDs terminados em :free: nunca troca para a versão paga sem você saber."""
    candidates = list(PREFERRED_MODELS)
    for mid in list_free_nemotron():
        if mid not in candidates:
            candidates.append(mid)
    skip = set(tried) | {load_model()}
    tests = 0
    for mid in candidates:
        if mid in skip or tests >= max_tests:
            continue
        tests += 1
        if test_api(key, mid)["ok"]:
            save_model(mid)
            write_opencode_config(mid)
            return mid
    return None


# ------------------------------------------------------------------ menu

PRIVACY = ("Modelos gratuitos podem registrar os prompts. Tudo o que a IA ler (código, strings,\n"
           "manifesto) é enviado pela internet. Use só em APK seu ou autorizado.")

PROMPT_SUMMARY = ("Você está numa pasta de um APK decodificado (apktool ou jadx). Leia o "
                  "AndroidManifest.xml e a estrutura de pastas e faça um resumo curto em português: "
                  "o que o app faz, permissões relevantes, componentes expostos e pontos que merecem "
                  "revisão manual. Não modifique nenhum arquivo.")
PROMPT_BUILD = ("Esta pasta é um APK decodificado pelo apktool e a reconstrução falhou com o erro abaixo. "
                "Explique a causa provável, aponte o arquivo e a linha e sugira a correção mínima. "
                "Não modifique nenhum arquivo, só explique.\n\nERRO:\n")


def ask_with_fallback(ui, folder: Path, prompt: str) -> dict[str, Any]:
    """ask_once; se o modelo gratuito saiu do ar (só acontece no opencode), troca por
    outro :free e tenta uma vez de novo."""
    res = ask_once(folder, prompt, emit=print)
    if res.get("model_unavailable") and load_provider() == PROVIDER_OPENCODE:
        key = load_key()
        print(ui.YELLOW + "Modelo gratuito indisponível. Procurando outro…" + ui.RESET)
        found = ensure_working_model(key, (load_model(),)) if key else None
        if found:
            print(ui.GREEN + f"Modelo trocado para openrouter/{found}. Tentando de novo…" + ui.RESET)
            res = ask_once(folder, prompt, emit=print)
        else:
            res["message"] += " Use [8] Trocar modelo para escolher outro."
    return res


def _choose_dir(ui) -> Path | None:
    outputs: Path = ui.OUTPUTS
    outputs.mkdir(parents=True, exist_ok=True)
    cands = sorted((p for p in outputs.iterdir()
                    if p.is_dir() and (p.name.endswith("_apktool") or p.name.endswith("_jadx"))),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:15]
    if cands:
        print(ui.CYAN + "Pastas disponíveis:" + ui.RESET)
        for i, item in enumerate(cands, 1):
            print(f" [{i}] {ui.sanitize(item.name)}")
        print(" [P] Informar outro caminho")
        choice = ui.safe_input("Escolha a pasta: ").strip()
        if choice.lower() != "p":
            try:
                return cands[int(choice) - 1]
            except (ValueError, IndexError):
                print(ui.YELLOW + "Seleção não reconhecida." + ui.RESET)
    raw = ui.safe_input("Caminho da pasta do APK decodificado: ").strip().strip('"').strip("'")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _paste_key(ui) -> str:
    print(ui.DIM + "Cole a chave do OpenRouter (a colagem não aparece na tela) e dê ENTER." + ui.RESET)
    try:
        return getpass.getpass("API key: ").strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return ui.safe_input("API key: ").strip()


def _setup(ui) -> bool:
    key = _paste_key(ui)
    if not key:
        print(ui.YELLOW + "Nada informado." + ui.RESET)
        return False
    result = configure(key)
    if not result["ok"]:
        print(ui.RED + result["message"] + ui.RESET)
        return False
    print(ui.GREEN + result["message"] + ui.RESET)
    print(f" Modelo: openrouter/{result['model']}")
    return _run_checks(ui)


def _run_checks_codex(ui) -> bool:
    print(ui.CYAN + "\nTestando (codex)…" + ui.RESET)
    binary = find_codex()
    print(f" [{ui.GREEN + 'OK' + ui.RESET if binary else ui.RED + 'falhou' + ui.RESET}] "
          f"codex: {binary or 'não encontrado no PATH'}")
    if not binary:
        return False
    print(ui.DIM + " Testando `codex exec` numa pasta neutra (até 2 min)… "
          "se sua versão usa outras flags, ajuste em ~/.readroid/ai.json (\"codex_exec_args\")." + ui.RESET)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    res = ask_once(WORKSPACE, "Responda apenas: OK. Não edite nenhum arquivo.", timeout=120, emit=out.append)
    print(f" [{ui.GREEN + 'OK' + ui.RESET if res['ok'] else ui.RED + 'falhou' + ui.RESET}] "
          f"codex exec: {res['message']}")
    if out:
        print(ui.DIM + "   " + " | ".join(out)[-160:] + ui.RESET)
    return bool(res["ok"])


def _run_checks(ui) -> bool:
    if load_provider() == PROVIDER_CODEX:
        return _run_checks_codex(ui)
    key, model = load_key(), load_model()
    if not key:
        print(ui.YELLOW + "Sem chave configurada." + ui.RESET)
        return False
    print(ui.CYAN + "\nTestando (opencode)…" + ui.RESET)
    api = test_api(key, model)
    if not api["ok"] and api.get("model_unavailable"):
        print(ui.YELLOW + f" Modelo {model} indisponível. Procurando outro gratuito…" + ui.RESET)
        found = ensure_working_model(key, (model,))
        if found:
            model = found
            api = {"ok": True, "message": f"Chave e modelo funcionando (trocado para {found})."}
    print(f" [{ui.GREEN + 'OK' + ui.RESET if api['ok'] else ui.RED + 'falhou' + ui.RESET}] "
          f"OpenRouter direto: {api['message']}")
    binary = find_opencode()
    print(f" [{ui.GREEN + 'OK' + ui.RESET if binary else ui.YELLOW + '--' + ui.RESET}] "
          f"opencode: {binary or 'não encontrado (instale o opencode para usar a IA)'}")
    if api["ok"] and binary:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        write_opencode_config(model)
        print(ui.DIM + " Testando o opencode numa pasta neutra (até 2 min)…" + ui.RESET)
        out: list[str] = []
        res = ask_once(WORKSPACE, "Responda apenas: OK", timeout=120, emit=out.append)
        print(f" [{ui.GREEN + 'OK' + ui.RESET if res['ok'] else ui.RED + 'falhou' + ui.RESET}] "
              f"opencode: {res['message']}")
        if out:
            print(ui.DIM + "   " + " | ".join(out)[-160:] + ui.RESET)
        return bool(res["ok"])
    return bool(api["ok"])


def _change_model(ui) -> None:
    print(ui.DIM + "Buscando modelos Nemotron gratuitos…" + ui.RESET)
    models = list_free_nemotron()
    if not models:
        print(ui.YELLOW + "Não consegui listar. Você pode digitar o ID manualmente." + ui.RESET)
    for i, mid in enumerate(models[:12], 1):
        print(f" [{i}] {ui.sanitize(mid)}")
    raw = ui.safe_input("Número, ID completo, ou ENTER para cancelar: ").strip()
    if not raw:
        return
    chosen = models[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(models) else raw
    if not re.match(r"^[\w.\-]+/[\w.\-:]+$", chosen):
        print(ui.RED + "ID inválido. Formato: fornecedor/modelo:free" + ui.RESET)
        return
    save_model(chosen)
    write_opencode_config(chosen)
    print(ui.GREEN + f"Modelo definido: openrouter/{chosen}" + ui.RESET)


# ------------------------------------------------------------------ relatórios e ciclo editar -> reconstruir

MAX_ROUNDS = 3
PROMPT_CROSS = ("Abaixo está o resultado da análise estática do ReaDroid para este APK (JSON resumido). "
                "Para cada URL, domínio, IP e permissão sensível, procure no código desta pasta onde é usado "
                "e avalie o risco. Responda em português, de forma curta e organizada. "
                "Não modifique nenhum arquivo.\n\nRELATÓRIOS:\n")
EDIT_RULES = ("Regras: edite SOMENTE arquivos dentro desta pasta; não altere apktool.yml; faça a menor "
              "mudança possível; não rode comandos de build (o ReaDroid reconstrói depois); ao final, liste "
              "os arquivos que alterou.\n\n")
RECREATE_RULES = ("Regras: trabalhe SOMENTE dentro desta pasta (saída do apktool/jadx); não altere "
                   "apktool.yml; não rode comandos de build (o ReaDroid reconstrói depois); faça as "
                   "mudanças em etapas pequenas e coerentes, sem quebrar o resto do app; ao final, liste "
                   "os arquivos que criou ou alterou e um resumo do que foi recriado.\n\n")
PROMPT_RECREATE = ("Esta pasta é um app Android decodificado (apktool/jadx). Primeiro leia o "
                    "AndroidManifest.xml e entenda a estrutura geral. Depois, recrie/reescreva a parte "
                    "pedida abaixo com código limpo e moderno, preservando o comportamento observável "
                    "(mesmas telas, mesmo fluxo, mesmos nomes de recursos usados pelo manifesto), a menos "
                    "que a instrução peça explicitamente para mudar o comportamento. Use só em um app seu "
                    "ou que você tem autorização para modificar.\n\nO QUE RECRIAR:\n")


def _shrink(obj: Any, max_items: int = 40, max_str: int = 200) -> Any:
    """Reduz um JSON de relatório para caber num prompt (listas e textos longos cortados)."""
    if isinstance(obj, dict):
        return {k: _shrink(v, max_items, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [_shrink(v, max_items, max_str) for v in obj[:max_items]]
        if len(obj) > max_items:
            out.append(f"... (+{len(obj) - max_items} itens omitidos)")
        return out
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    return obj


def find_reports(reports_dir: Path, decoded: Path) -> list[Path]:
    """Relatório mais recente de cada tipo para o APK desta pasta (nome: <apk>_<tipo>_<data>.json)."""
    stem = re.sub(r"_(apktool|jadx)$", "", decoded.name)
    latest: dict[str, Path] = {}
    try:
        files = sorted(reports_dir.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return []
    for path in files:
        kind = path.name[len(stem) + 1:].rsplit("_", 2)[0]
        latest[kind] = path  # os mais novos vêm por último e sobrescrevem
    return list(latest.values())


def build_report_context(reports_dir: Path, decoded: Path, limit: int = 6000) -> str:
    chunks = []
    for path in find_reports(reports_dir, decoded):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        chunks.append(f"## {path.name}\n" + json.dumps(_shrink(data), ensure_ascii=False))
    return "\n\n".join(chunks)[:limit]


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for item in root.rglob("*"):
        try:
            if item.is_file():
                st = item.stat()
                snap[str(item.relative_to(root))] = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
    return snap


def diff_snapshots(before: dict, after: dict) -> dict[str, list[str]]:
    return {"changed": sorted(k for k in after if k in before and before[k] != after[k]),
            "added": sorted(k for k in after if k not in before),
            "removed": sorted(k for k in before if k not in after)}


def build_error_text(result: dict[str, Any]) -> str:
    steps = result.get("steps", {})
    for name in ("quick_edits", "rebuild", "align", "keystore", "sign", "verify"):
        step = steps.get(name)
        if step and not step.get("ok") and not step.get("skipped"):
            return (f"Etapa: {name}\n{step.get('message', '')}\n"
                    f"{str(step.get('stderr', ''))[-3000:]}\n{str(step.get('stdout', ''))[-1000:]}").strip()
    return str(result.get("message", "Falha desconhecida na reconstrução."))


def fix_loop(decoded: Path, instruction: str, build_fn, ask_fn, confirm_fn, say,
             max_rounds: int = MAX_ROUNDS, rules: str = EDIT_RULES) -> dict[str, Any]:
    """Ciclo: (IA edita) -> reconstrói -> se falhar, devolve o erro à IA -> reconstrói de novo.

    build_fn() -> resultado do build_pipeline; ask_fn(prompt) -> {"ok":..,"message":..};
    confirm_fn(texto) -> bool; say(texto) imprime; rules -> cabeçalho de regras prefixado em
    cada chamada (EDIT_RULES para edição pontual, RECREATE_RULES para recriação). Tudo
    injetado para ser testável."""
    def report_changes(before):
        d = diff_snapshots(before, snapshot(decoded))
        n = sum(len(v) for v in d.values())
        say(f"Arquivos mexidos pela IA: {n} (alterados {len(d['changed'])}, novos {len(d['added'])}, "
            f"removidos {len(d['removed'])})")
        for name in (d["changed"] + d["added"] + d["removed"])[:15]:
            say("   " + name)

    if instruction:
        before = snapshot(decoded)
        res = ask_fn(rules + f"Pasta: {decoded}\nTarefa: {instruction}")
        if not res.get("ok"):
            return {"ok": False, "rounds": 0, "message": res.get("message", "A IA não respondeu.")}
        report_changes(before)
    last: dict[str, Any] = {}
    for rnd in range(1, max_rounds + 1):
        say(f"Reconstruindo (tentativa {rnd}/{max_rounds})…")
        last = build_fn()
        if last.get("ok"):
            return {"ok": True, "rounds": rnd, "result": last, "message": "APK reconstruído."}
        error = build_error_text(last)
        say("Falhou:\n" + error[-600:])
        if rnd == max_rounds or not confirm_fn("Pedir para a IA corrigir e tentar de novo?"):
            break
        before = snapshot(decoded)
        res = ask_fn(rules + f"Pasta: {decoded}\nA reconstrução falhou com este erro. Corrija a causa:\n{error}")
        if not res.get("ok"):
            return {"ok": False, "rounds": rnd, "result": last, "message": res.get("message", "A IA não respondeu.")}
        report_changes(before)
    return {"ok": False, "rounds": rnd if last else 0, "result": last, "message": "Não foi possível reconstruir."}


def _print_build(ui, result: dict[str, Any]) -> None:
    for name, label in (("quick_edits", "Edições"), ("rebuild", "Reconstrução"), ("align", "Alinhamento"),
                        ("keystore", "Chave"), ("sign", "Assinatura"), ("verify", "Verificação")):
        step = result.get("steps", {}).get(name)
        if step:
            mark = (ui.GREEN + "OK" if step.get("ok") else ui.YELLOW + "pulado" if step.get("skipped")
                    else ui.RED + "falhou") + ui.RESET
            print(f" [{mark}] {label}")


def _backup(ui, decoded: Path) -> Path | None:
    dest = ui.BUILDS / f"_backup_{decoded.name}_{__import__('datetime').datetime.now():%Y%m%d_%H%M%S}"
    try:
        shutil.copytree(decoded, dest)
        return dest
    except (OSError, shutil.Error):
        return None


def _edit_and_rebuild(ui, decoded: Path) -> None:
    from . import builder
    from datetime import datetime
    if not (decoded / "apktool.yml").is_file():
        print(ui.YELLOW + "Essa pasta não é uma saída do Apktool (opção 5 do menu principal)." + ui.RESET)
        return
    print(ui.YELLOW + "Use só em APK seu ou que você tem autorização para modificar." + ui.RESET)
    instruction = ui.safe_input("O que a IA deve fazer? (ENTER = só reconstruir e corrigir erros): ").strip()
    print(f"A IA vai editar arquivos em {decoded}. Será feito um backup antes.")
    if ui.safe_input("Continuar? (s/N): ").strip().lower() != "s":
        return
    backup = _backup(ui, decoded)
    if backup is None:
        print(ui.YELLOW + "Não consegui criar o backup (espaço?)." + ui.RESET)
        if ui.safe_input("Continuar sem backup? (s/N): ").strip().lower() != "s":
            return
    else:
        print(ui.GREEN + f"Backup: {backup}" + ui.RESET)

    def build_fn():
        name = decoded.name.replace("_apktool", "") or "app"
        work = ui.BUILDS / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        result = builder.build_pipeline(decoded, work, ui.KEYSTORE, builder.DEFAULT_ALIAS,
                                        builder.DEFAULT_STOREPASS, [], lambda m: print(ui.CYAN + " · " + m + ui.RESET))
        _print_build(ui, result)
        return result

    out = fix_loop(decoded, instruction, build_fn, lambda prompt: ask_with_fallback(ui, decoded, prompt),
                   lambda q: ui.safe_input(q + " (s/N): ").strip().lower() == "s", print)
    if out["ok"]:
        print(ui.GREEN + f"\nAPK pronto: {out['result'].get('final_apk')}" + ui.RESET)
    else:
        print(ui.RED + "\n" + out["message"] + ui.RESET)
        if backup:
            print(f"Para desfazer as mudanças da IA, restaure o backup: {backup}")


def _recreate_with_ai(ui, decoded: Path) -> None:
    """Recria/reescreve uma parte do app do zero (ou o app inteiro, módulo por módulo).
    Reaproveita o mesmo ciclo editar -> reconstruir -> corrigir do _edit_and_rebuild, só que
    com um prompt de recriação em vez de edição pontual. Pensado para o provedor codex
    (modo automático, várias etapas sozinho), mas funciona com qualquer provedor ativo."""
    from . import builder
    from datetime import datetime
    if not (decoded / "apktool.yml").is_file():
        print(ui.YELLOW + "Essa pasta não é uma saída do Apktool (opção 5 do menu principal)." + ui.RESET)
        return
    provider = load_provider()
    print(ui.YELLOW + "Recriação de código é uma mudança grande. Use só em APK seu ou autorizado." + ui.RESET)
    print(ui.DIM + f"Provedor ativo: {PROVIDER_LABELS[provider]}" + ui.RESET)
    target = ui.safe_input(
        "O que recriar? (ex.: 'a tela de login', 'o parser de XML', "
        "'tudo, modernizando o código') : "
    ).strip()
    if not target:
        print(ui.YELLOW + "Nada informado." + ui.RESET)
        return
    print(f"A IA vai reescrever arquivos em {decoded}. Será feito um backup antes.")
    if ui.safe_input("Continuar? (s/N): ").strip().lower() != "s":
        return
    backup = _backup(ui, decoded)
    if backup is None:
        print(ui.YELLOW + "Não consegui criar o backup (espaço?)." + ui.RESET)
        if ui.safe_input("Continuar sem backup? (s/N): ").strip().lower() != "s":
            return
    else:
        print(ui.GREEN + f"Backup: {backup}" + ui.RESET)

    def build_fn():
        name = decoded.name.replace("_apktool", "") or "app"
        work = ui.BUILDS / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        result = builder.build_pipeline(decoded, work, ui.KEYSTORE, builder.DEFAULT_ALIAS,
                                        builder.DEFAULT_STOREPASS, [], lambda m: print(ui.CYAN + " · " + m + ui.RESET))
        _print_build(ui, result)
        return result

    instruction = PROMPT_RECREATE + target
    out = fix_loop(decoded, instruction, build_fn, lambda prompt: ask_with_fallback(ui, decoded, prompt),
                   lambda q: ui.safe_input(q + " (s/N): ").strip().lower() == "s", print,
                   rules=RECREATE_RULES)
    if out["ok"]:
        print(ui.GREEN + f"\nAPK recriado e reconstruído: {out['result'].get('final_apk')}" + ui.RESET)
    else:
        print(ui.RED + "\n" + out["message"] + ui.RESET)
        if backup:
            print(f"Para desfazer as mudanças da IA, restaure o backup: {backup}")


def _choose_provider(ui) -> None:
    current = load_provider()
    print(ui.CYAN + "\nProvedores disponíveis:" + ui.RESET)
    for i, prov in enumerate(PROVIDERS, 1):
        mark = " (atual)" if prov == current else ""
        print(f" [{i}] {PROVIDER_LABELS[prov]}{mark}")
    raw = ui.safe_input("Escolha o provedor (ENTER para cancelar): ").strip()
    if not raw:
        return
    try:
        chosen = PROVIDERS[int(raw) - 1]
    except (ValueError, IndexError):
        print(ui.YELLOW + "Opção inválida." + ui.RESET)
        return
    save_provider(chosen)
    print(ui.GREEN + f"Provedor ativo: {PROVIDER_LABELS[chosen]}" + ui.RESET)
    if chosen == PROVIDER_CODEX and not find_codex():
        print(ui.YELLOW + "Aviso: não achei o binário `codex` no PATH agora. Confirme que ele "
              "funciona direto no Termux (fora do ReaDroid) antes de usar." + ui.RESET)
    if chosen == PROVIDER_OPENCODE and load_key() is None:
        print(ui.YELLOW + "Aviso: o provedor opencode precisa de uma chave do OpenRouter "
              "(opção [10] Trocar a chave)." + ui.RESET)


def run_menu(ui) -> None:
    """Menu 13. `ui` é o módulo app (cores, safe_input, sanitize, OUTPUTS, REPORTS, BUILDS, KEYSTORE)."""
    print(ui.CYAN + "\nASSISTENTE DE IA (opcional)" + ui.RESET)
    print(ui.DIM + "O ReaDroid funciona normalmente sem isto. Ignore se não quiser usar IA." + ui.RESET)
    if load_provider() == PROVIDER_OPENCODE and load_key() is None:
        print("Nenhuma chave configurada ainda para o opencode.")
        print(ui.DIM + "Se você usa o Codex (já configurado no Termux), escolha [P] no menu abaixo." + ui.RESET)
        if ui.safe_input("Configurar o opencode agora? Basta colar a API key do OpenRouter (s/N): ").strip().lower() == "s":
            _setup(ui)
    _menu_loop(ui)


def _menu_loop(ui) -> None:
    consented = False
    while True:
        provider = load_provider()
        status = f"openrouter/{load_model()}" if provider == PROVIDER_OPENCODE else "binário global do Termux"
        print(f"\n Provedor: {PROVIDER_LABELS[provider]}  ({status})")
        print(" [1] Abrir o assistente numa pasta de APK (interativo)")
        print(" [2] Resumir o app")
        print(" [3] Cruzar relatórios do ReaDroid com o código")
        print(" [4] Editar com a IA e reconstruir (corrige erros em ciclo)")
        print(" [5] Recriar um trecho do app do zero (automático, ciclo completo)")
        print(" [6] Explicar um erro de reconstrução")
        print(" [7] Pergunta livre sobre o APK")
        print(" [8] Testar / diagnosticar")
        print(" [9] Trocar modelo (só opencode)")
        print(" [10] Trocar a chave (só opencode)")
        print(" [P] Trocar provedor de IA (opencode/codex)")
        print(" [11] Desativar IA (apaga chave e config)")
        print(" [0] Voltar")
        choice = ui.safe_input("IA > ").strip()
        if choice in ("", "0"):
            return
        low = choice.lower()
        if low == "p":
            _choose_provider(ui)
            continue
        if choice in ("1", "2", "3", "4", "5", "6", "7"):
            if not consented:
                print(ui.YELLOW + PRIVACY + ui.RESET)
                if ui.safe_input("Continuar? (s/N): ").strip().lower() != "s":
                    continue
                consented = True
            folder = _choose_dir(ui)
            if folder is None:
                print(ui.YELLOW + "Pasta não encontrada." + ui.RESET)
                continue
            res: dict[str, Any] | None = None
            if choice == "1":
                res = open_interactive(folder)
            elif choice == "2":
                res = ask_with_fallback(ui, folder, PROMPT_SUMMARY)
            elif choice == "3":
                ctx = build_report_context(ui.REPORTS, folder)
                if not ctx:
                    print(ui.YELLOW + "Sem relatórios para este APK. Rode antes as opções 1, 2 e 3 do menu principal." + ui.RESET)
                    continue
                res = ask_with_fallback(ui, folder, PROMPT_CROSS + ctx)
            elif choice == "4":
                _edit_and_rebuild(ui, folder)
            elif choice == "5":
                _recreate_with_ai(ui, folder)
            elif choice == "6":
                err = ui.safe_input("Cole o erro (uma linha) ou ENTER para cancelar: ").strip()
                if not err:
                    continue
                res = ask_with_fallback(ui, folder, PROMPT_BUILD + err[:4000])
            else:
                question = ui.safe_input("Sua pergunta: ").strip()
                if not question:
                    continue
                res = ask_with_fallback(ui, folder, question[:4000])
            if res is not None:
                print((ui.GREEN if res["ok"] else ui.YELLOW) + res["message"] + ui.RESET)
        elif choice == "8":
            _run_checks(ui)
        elif choice == "9":
            _change_model(ui)
        elif choice == "10":
            _setup(ui)
        elif choice == "11":
            if ui.safe_input("Apagar chave e configuração da IA (opencode e codex)? (s/N): ").strip().lower() == "s":
                gone = disable()
                print(ui.GREEN + ("Removido: " + ", ".join(gone) if gone else "Nada para remover.") + ui.RESET)
                return
        else:
            print(ui.YELLOW + "Opção inválida." + ui.RESET)
