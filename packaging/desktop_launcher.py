"""Desktop launcher for the packaged macOS app.

Responsibilities that only make sense for a double-click desktop app (none of
this applies to the normal `uvicorn app.main:app` server path used for local
dev / Docker / cloud deployments):

- Uses a per-user writable data directory under ~/Library/Application Support
  instead of a path inside the (read-only, potentially code-signed) .app
  bundle.
- Persists a stable SECRET_KEY across restarts, generated once on first run,
  so logging in survives quitting and reopening the app.
- Picks an open local port, starts the server, and shows the dashboard in a
  native app window (WKWebView) once it's actually ready -- not the user's
  default browser, so it behaves like a real Mac app instead of hijacking
  Safari/Chrome every launch.

This file is the PyInstaller entry point; it is not imported by the normal
FastAPI app and has no effect on any other deployment path.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import sys
import threading
from pathlib import Path

APP_SUPPORT_DIR_NAME = "Conversation Intelligence"


def _bundled_resource_path(relative: str) -> Path:
    """Path to a bundled resource -- works both frozen (PyInstaller) and
    unfrozen (running this script directly during development/testing)."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))
    else:
        base = Path(__file__).resolve().parent.parent
    return base / relative


def _setup_data_directory() -> Path:
    data_dir = Path.home() / "Library" / "Application Support" / APP_SUPPORT_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)

    seed_source = _bundled_resource_path("data/knowledge_base.xlsx")
    seed_dest = data_dir / "knowledge_base.xlsx"
    if seed_source.exists() and not seed_dest.exists():
        shutil.copy2(seed_source, seed_dest)

    return data_dir


def _setup_secret_key(data_dir: Path) -> None:
    key_file = data_dir / ".secret_key"
    if key_file.exists():
        key = key_file.read_text().strip()
    else:
        key = secrets.token_urlsafe(32)
        key_file.write_text(key)
        key_file.chmod(0o600)
    os.environ["SECRET_KEY"] = key


def _find_open_port(preferred: int = 8000) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def _build_webview_delegate_class():
    """Builds the WKUIDelegate/WKNavigationDelegate PyObjC subclass lazily,
    inside a function, so this module still imports cleanly off-macOS (e.g.
    in CI running the plain pytest suite, where AppKit/WebKit don't exist).

    A bare WKWebView does NOT get browser-grade behavior for free -- unlike
    Safari/Chrome, it has no built-in file picker for <input type=file>, no
    JS alert()/confirm() dialogs, and no automatic "Save As" for downloads
    (the Export CSV/Excel buttons). Each of those requires the host app to
    implement the corresponding delegate callback; without them the web
    content just does nothing when e.g. the Upload button is clicked, which
    is exactly the "upload doesn't work" symptom this class fixes.
    """
    from AppKit import (
        NSAlert,
        NSModalResponseOK,
        NSOpenPanel,
        NSSavePanel,
    )
    from Foundation import NSObject
    from WebKit import WKNavigationResponsePolicyAllow, WKNavigationResponsePolicyDownload

    class _WebViewDelegate(NSObject):
        # -- WKUIDelegate: file picker for <input type="file"> --------
        #
        # `panel` and the response closure previously lived only as local
        # variables inside this method -- PyObjC's bridge holds ObjC-side
        # blocks weakly, so nothing guaranteed the Python objects survived
        # until the user actually finished interacting with the sheet. That
        # produced exactly the "works the first time, silently does nothing
        # after that until the app is restarted" symptom: intermittent GC of
        # the panel/callback between the panel being shown and the user's
        # click resolving it. Stashing them on `self` (which lives for the
        # whole app process) removes that race entirely.
        def webView_runOpenPanelWithParameters_initiatedByFrame_completionHandler_(
            self, webView, parameters, frame, completionHandler
        ):
            panel = NSOpenPanel.openPanel()
            panel.setCanChooseFiles_(True)
            panel.setCanChooseDirectories_(False)
            try:
                panel.setAllowsMultipleSelection_(parameters.allowsMultipleSelection())
            except Exception:
                pass

            self._pending_open_panel = panel
            self._pending_open_completion = completionHandler

            window = webView.window()
            if window is not None:
                window.makeKeyAndOrderFront_(None)
                panel.beginSheetModalForWindow_completionHandler_(
                    window, self._handleOpenPanelResponse_
                )
            else:
                self._handleOpenPanelResponse_(panel.runModal())

        def _handleOpenPanelResponse_(self, response):
            panel = self._pending_open_panel
            completionHandler = self._pending_open_completion
            self._pending_open_panel = None
            self._pending_open_completion = None
            try:
                if response == NSModalResponseOK and panel is not None:
                    completionHandler(panel.URLs())
                else:
                    completionHandler(None)
            except Exception:
                pass

        # -- WKUIDelegate: JS alert()/confirm() ------------------------
        def webView_runJavaScriptAlertPanelWithMessage_initiatedByFrame_completionHandler_(
            self, webView, message, frame, completionHandler
        ):
            alert = NSAlert.alloc().init()
            alert.setMessageText_(str(message))
            alert.addButtonWithTitle_("OK")
            alert.runModal()
            completionHandler()

        def webView_runJavaScriptConfirmPanelWithMessage_initiatedByFrame_completionHandler_(
            self, webView, message, frame, completionHandler
        ):
            alert = NSAlert.alloc().init()
            alert.setMessageText_(str(message))
            alert.addButtonWithTitle_("OK")
            alert.addButtonWithTitle_("Cancel")
            response = alert.runModal()
            completionHandler(response == 1000)  # NSAlertFirstButtonReturn

        # -- WKNavigationDelegate: route file-download responses -------
        # (Export CSV/Excel navigate the webview to a Content-Disposition:
        # attachment response -- without this, WKWebView just tries to
        # render the file inline instead of downloading it.)
        def webView_decidePolicyForNavigationResponse_decisionHandler_(
            self, webView, navigationResponse, decisionHandler
        ):
            response = navigationResponse.response()
            is_download = False
            try:
                headers = response.allHeaderFields()
                disposition = headers.get("Content-Disposition", "") or headers.get(
                    "content-disposition", ""
                )
                if disposition and "attachment" in disposition.lower():
                    is_download = True
            except Exception:
                pass
            if not navigationResponse.canShowMIMEType():
                is_download = True
            decisionHandler(
                WKNavigationResponsePolicyDownload if is_download else WKNavigationResponsePolicyAllow
            )

        def webView_navigationResponse_didBecomeDownload_(self, webView, navigationResponse, download):
            download.setDelegate_(self)

        def webView_navigationAction_didBecomeDownload_(self, webView, navigationAction, download):
            download.setDelegate_(self)

        # -- WKDownloadDelegate: ask where to save ----------------------
        def download_decideDestinationUsingResponse_suggestedFilename_completionHandler_(
            self, download, response, suggestedFilename, completionHandler
        ):
            panel = NSSavePanel.savePanel()
            panel.setNameFieldStringValue_(str(suggestedFilename))
            result = panel.runModal()
            if result == NSModalResponseOK and panel.URL() is not None:
                completionHandler(panel.URL())
            else:
                completionHandler(None)

    return _WebViewDelegate


class _AppWindow:
    """A native window hosting a WKWebView pointed at the local server --
    this is what makes the app feel like a real Mac app (its own window,
    Cmd+W closes it without quitting, Dock icon reopens it) instead of a
    background process that just launches Safari."""

    _LOADING_HTML = (
        "<html><body style='display:flex;align-items:center;justify-content:center;"
        "height:100vh;margin:0;font-family:-apple-system;background:#0f172a;color:#fff;'>"
        "<p>Starting Conversation Intelligence&hellip;</p></body></html>"
    )

    def __init__(self, url: str) -> None:
        from AppKit import (
            NSApp,
            NSBackingStoreBuffered,
            NSMakeRect,
            NSWindow,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskMiniaturizable,
            NSWindowStyleMaskResizable,
            NSWindowStyleMaskTitled,
        )
        from Foundation import NSURL, NSURLRequest
        from WebKit import WKWebView

        self._url = url
        self._NSURL = NSURL
        self._NSURLRequest = NSURLRequest
        self._NSApp = NSApp

        style_mask = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        rect = NSMakeRect(0, 0, 1280, 860)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style_mask, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Conversation Intelligence")
        self.window.setMinSize_((900, 600))
        self.window.center()
        # Keep the NSWindow alive across close/reopen instead of it being
        # deallocated when the user clicks the red close button.
        self.window.setReleasedWhenClosed_(False)

        self.webview = WKWebView.alloc().initWithFrame_(rect)

        # Keep a strong reference on self -- PyObjC delegate properties are
        # weak/unretained from the Cocoa side, so without this the Python
        # object would get garbage-collected right after __init__ returns
        # and the delegate methods would silently stop firing.
        delegate_cls = _build_webview_delegate_class()
        self._delegate = delegate_cls.alloc().init()
        self.webview.setUIDelegate_(self._delegate)
        self.webview.setNavigationDelegate_(self._delegate)

        self.webview.loadHTMLString_baseURL_(self._LOADING_HTML, None)
        self.window.setContentView_(self.webview)

    def load_dashboard(self) -> None:
        request = self._NSURLRequest.requestWithURL_(self._NSURL.URLWithString_(self._url))
        self.webview.loadRequest_(request)

    def show(self) -> None:
        self.window.makeKeyAndOrderFront_(None)
        self._NSApp.activateIgnoringOtherApps_(True)


class _MenuBarApp:
    """A small always-on menu bar icon (rumps -> real Cocoa/NSStatusBar
    integration) so the app has a visible presence and a proper Quit,
    instead of being an invisible background process the user has to hunt
    down in Activity Monitor to stop."""

    def __init__(self, port: int, data_dir: Path) -> None:
        import rumps

        self._rumps = rumps
        self.port = port
        self.data_dir = data_dir
        self.window = _AppWindow(f"http://127.0.0.1:{port}/ui")
        self.app = rumps.App("Conversation Intelligence", title="CI", quit_button="Quit")
        self.app.menu = [
            rumps.MenuItem("Open Dashboard", callback=self._open_dashboard),
            rumps.MenuItem("Show Data Folder", callback=self._show_data_folder),
        ]
        # Poll on the app's own run loop (main-thread-safe) until the
        # server responds, then load the real dashboard and show the
        # window -- avoids a "can't connect" flash in the WebView.
        self._ready = False
        self._ready_timer = rumps.Timer(self._check_ready, 0.4)
        self._ready_timer.start()

    def _check_ready(self, _timer) -> None:
        if self._ready:
            return
        import urllib.request

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
        except Exception:
            return
        self._ready = True
        self._ready_timer.stop()
        self.window.load_dashboard()
        self.window.show()

    def _open_dashboard(self, _sender) -> None:
        self.window.show()

    def _show_data_folder(self, _sender) -> None:
        import subprocess

        subprocess.run(["open", str(self.data_dir)])

    def run(self) -> None:
        self.app.run()


def main() -> None:
    if not getattr(sys, "frozen", False):
        # Dev/test convenience: running this script directly puts
        # packaging/ on sys.path, not the repo root where the `app`
        # package lives. PyInstaller's frozen bundle already has `app`
        # importable, so this branch is a no-op there.
        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    data_dir = _setup_data_directory()
    os.environ["KNOWLEDGE_BASE_FILE"] = str(data_dir / "knowledge_base.xlsx")
    os.environ.setdefault("LOG_JSON", "false")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    _setup_secret_key(data_dir)

    # Point at the embedding model bundled inside the app instead of the
    # Hugging Face Hub ID, so recipients never need network access to load
    # it (and never hit a slow/failing first-run download).
    bundled_model = _bundled_resource_path("packaging/bundled_model/all-MiniLM-L6-v2")
    if bundled_model.exists():
        os.environ["EMBEDDING_MODEL_NAME"] = str(bundled_model)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Imported only after the env vars above are set -- app.core.config's
    # `settings` is a module-level singleton read once at import time.
    import uvicorn

    from app.main import app

    port = _find_open_port(8000)

    # The server runs on a daemon thread; the main thread is handed to
    # rumps' Cocoa event loop below (rumps.App.run() must own the main
    # thread on macOS). Quitting from the menu bar terminates the process,
    # which takes the daemon thread down with it -- no separate shutdown
    # handshake needed.
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=app, host="127.0.0.1", port=port, log_level="info", workers=1),
        daemon=True,
    )
    server_thread.start()

    print(f"Conversation Intelligence starting at http://127.0.0.1:{port}/ui")
    print(f"Data stored in: {data_dir}")
    _MenuBarApp(port, data_dir).run()


def _show_crash_alert(message: str) -> None:
    """Best-effort native alert dialog -- the packaged app runs windowed
    (no console), so a silent crash on startup would otherwise be
    invisible to a non-technical user."""
    try:
        import subprocess

        script = f'display alert "Conversation Intelligence failed to start" message "{message}" as critical'
        subprocess.run(["osascript", "-e", script], timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- last-resort visibility for a windowed app
        _show_crash_alert(str(exc).replace('"', "'")[:500])
        raise
