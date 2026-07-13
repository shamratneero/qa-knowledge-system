# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the macOS desktop build.

Build with:
    pyinstaller packaging/desktop_app.spec --noconfirm

Produces dist/ConversationIntelligence.app, which packaging/build_dmg.sh
then wraps into a distributable .dmg.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

REPO_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 -- SPECPATH is injected by PyInstaller

datas = []
binaries = []
hiddenimports = []

# These packages do dynamic imports / ship non-Python data files that
# PyInstaller's static analysis can't see on its own.
for pkg in (
    "sentence_transformers",
    "torch",
    "sklearn",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "rumps",
    "AppKit",
    "Foundation",
    "objc",
    "WebKit",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# App static assets, seed Q&A knowledge base, and the bundled (offline)
# embedding model -- destination paths here are what
# packaging/desktop_launcher.py's _bundled_resource_path() looks for.
datas += [
    (str(REPO_ROOT / "app" / "static"), "app/static"),
    (str(REPO_ROOT / "data" / "knowledge_base.xlsx"), "data"),
    (
        str(REPO_ROOT / "packaging" / "bundled_model" / "all-MiniLM-L6-v2"),
        "packaging/bundled_model/all-MiniLM-L6-v2",
    ),
]

hiddenimports += [
    "app.main",
    "app.api.auth",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "jwt",
    "openpyxl",
    "multipart",
]

a = Analysis(
    [str(REPO_ROOT / "packaging" / "desktop_launcher.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ConversationIntelligence",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # server + menu bar app; no dangling Terminal window needed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ConversationIntelligence",
)

app = BUNDLE(
    coll,
    name="ConversationIntelligence.app",
    icon=None,
    bundle_identifier="com.conversationintelligence.desktop",
    info_plist={
        "NSHighResolutionCapable": "True",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleName": "Conversation Intelligence",
        # A regular foreground app: shows both a Dock icon AND the menu
        # bar status item (LSUIElement=True would hide the Dock icon --
        # not what's wanted here). LSBackgroundOnly is the fully headless
        # "agent" app type with NO UI presence at all; must stay False or
        # PyInstaller's default breaks both the Dock icon and menu bar.
        "LSUIElement": False,
        "LSBackgroundOnly": False,
        # Defensive: loopback (127.0.0.1) is exempt from App Transport
        # Security by default, but be explicit so the embedded WKWebView
        # never fails to load the local dashboard over plain HTTP.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    },
)
