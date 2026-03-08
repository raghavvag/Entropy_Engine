# Loom Video Script — hotreload CLI Tool

> **Total Duration:** 8–10 minutes
> **Recording Tool:** Loom (screen + face cam)
> **Setup:** VS Code open with the project, two PowerShell terminals side by side

---

## PRE-RECORDING CHECKLIST

Run these commands before hitting record:

```powershell
cd d:\HotReload_internship
Stop-Process -Name server -Force -ErrorAction SilentlyContinue
Stop-Process -Name hotreload -Force -ErrorAction SilentlyContinue
go build -o bin/hotreload.exe ./cmd/hotreload
```

- Make sure `samples/testserver/main.go` has the original `"hello world"` string
- Close unnecessary VS Code tabs
- Open two terminals: **Terminal 1** (for hotreload) and **Terminal 2** (for curl / edits)
- Have the GitHub repo page open in a browser tab
- Set VS Code font size large enough for recording (Ctrl + = to zoom in)

---

## SECTION 1 — INTRO (0:00 – 0:45)

### What to say:
> I am Raghav Agrawal, a second-year B.Tech Computer Science and Engineering student at Vellore Institute of Technology (VIT), Vellore. I am a backend-focused developer with strong experience in Java, Spring Boot, and distributed backend systems.

 I built a tool called hotreload — it's a CLI tool written in Go that watches your project folder for code changes and automatically rebuilds and restarts the server. So as a developer, you just save your file and the server restarts with the new code in under 2 seconds. No more manual stop, build, run cycle."

### What to show on screen:
- **Browser → GitHub repo page** (`https://github.com/raghavvag/go-hotreload-engine`)
- Scroll the **README.md** briefly — show the features list, the architecture diagram, the flags table
- Point out the **CI badge** (green checkmark — tests pass on Ubuntu + Windows)

### What to say:
> "Let me walk you through the architecture, show you the code, demonstrate all the features live, and then run the tests."

---

## SECTION 2 — ARCHITECTURE OVERVIEW (0:45 – 2:30)

### What to say:
> "The tool has a clean pipeline architecture. Let me show you the flow."

### What to show on screen:
- **VS Code → open `README.md`** — scroll to the **"How It Works"** section with the ASCII diagram

### What to say (walk through the diagram):
> "The pipeline is: Watcher → Debouncer → Builder → Runner. Each component communicates through Go channels — there's no shared mutable state. Let me explain each one."

---

### 2a — Watcher

**Open file:** `internal/watcher/watcher.go`

**What to say:**
> "The Watcher uses the fsnotify library to recursively watch all directories under the project root. At startup, it walks the entire directory tree and adds a watch on every non-ignored directory."

**Scroll to `addRecursive()` function (line ~87-102):**
> "Here in `addRecursive`, it uses `filepath.WalkDir` to traverse all directories. For each directory, it checks against our ignore matcher — if it's something like `.git` or `node_modules`, we skip it entirely."

**Scroll to `handleEvent()` function (line ~157-193):**
> "When a file event comes in — a create, write, remove, or rename — `handleEvent` processes it. If a new directory is created, we automatically start watching it recursively. If a directory is deleted, we clean up the watch. This is important because projects are dynamic — developers create and delete folders all the time."

**Scroll to `NewWithLimit()` (line ~35-78):**
> "We also have a scalability guard. Operating systems limit how many files you can watch — Linux has the inotify limit. If the watch count exceeds the configured `--max-watches` limit, we log a warning with hints on how to increase it."

---

### 2b — Ignore Matcher

**Open file:** `internal/watcher/ignore.go`

**What to say:**
> "The ignore system is important because a typical project has many files the tool shouldn't care about."

**Scroll to `defaultIgnorePatterns` and `defaultIgnoreExtensions` (line ~11-34):**
> "By default, we ignore `.git`, `node_modules`, `bin`, `vendor`, IDE configuration directories, plus all editor temp files like `.swp`, `.tmp`, `.bak`. These are the files Vim, Emacs, and VS Code create when you save."

**Scroll to `ShouldIgnore()` function (line ~83-139):**
> "The `ShouldIgnore` function checks every path against directory patterns, file extension patterns, editor backup patterns like tilde files and emacs lock files, glob patterns, and finally the extension whitelist. If you pass `--extensions '.go'`, only `.go` file changes trigger rebuilds — everything else is silently ignored."

**Scroll to `LoadHotreloadIgnore()` (line ~163-181):**
> "We also support a `.hotreloadignore` file — similar to `.gitignore`. You place it in your project root, one pattern per line, and it gets merged with the defaults."

---

### 2c — Debouncer

**Open file:** `internal/debounce/debounce.go`

**What to say:**
> "The debouncer is critical. When you hit save in VS Code, the editor might trigger 3-5 filesystem events — write the file, create a temp file, rename, etc. Without debouncing, you'd get 5 unnecessary rebuilds."

**Scroll to `loop()` function (line ~39-81):**
> "The logic is timer-based. When the first event arrives, we start a timer — default 300 milliseconds. Each new event resets the timer. Only after 300ms of complete silence do we emit a single trigger to the builder. So 50 rapid events produce just one rebuild."

---

### 2d — Builder

**Open file:** `internal/builder/builder.go`

**What to say:**
> "The builder receives triggers from the debouncer and runs the build command."

**Scroll to `runBuild()` function (line ~69-114):**
> "The key design decision here is cancellation. We use `context.WithCancel`. If a new file change arrives while a build is still running, we cancel the old build immediately and start a fresh one. This means only the latest code state gets built — you never wait for a stale build to finish."

> "When the build fails — say you have a syntax error — the compiler output is captured and logged with `slog.Error`. Importantly, the builder does NOT signal the runner. So the old server keeps running with the last good code. You never lose your running server because of a typo."

> "Only on success do we send a signal on the `Success` channel to tell the runner to restart."

---

### 2e — Runner

**Open file:** `internal/runner/runner.go`

**What to say:**
> "The runner manages the server process lifecycle."

**Scroll to `start()` function (line ~138-198):**
> "When starting the server, we call `utils.SetupProcessGroup` — this runs the process in its own process group. We get stdout and stderr pipes and stream them line by line using `bufio.Scanner` — this is real-time streaming, not buffered output."

**Scroll to `stop()` function (line ~103-133):**
> "When stopping, we call `KillProcessGroupWithGrace` which kills the entire process tree — not just the parent. This is important because your server might spawn child goroutines or sub-processes. On Unix, we send SIGTERM to the process group, wait a grace period, then SIGKILL. On Windows, we use `taskkill /T /F` to kill the tree."

**Scroll to `recordCrash()` function (line ~210-243):**
> "We also have crash-loop detection. If the server crashes more than 5 times in 30 seconds, we pause auto-restart and log an error telling the developer to fix their code. This prevents the tool from spinning in an infinite crash-restart loop. When you save a new file change, the pause resets automatically."

---

### 2f — Platform-Specific Kill

**Open file:** `internal/utils/kill_unix.go`

**What to say:**
> "For cross-platform support, we use Go build tags."

> "On Unix — `kill_unix.go` — we set `Setpgid: true` when starting the process, which gives it its own process group. To kill, we send SIGTERM to the negative PGID, which targets the entire group. If the process doesn't exit within the grace period, we escalate to SIGKILL."

**Open file:** `internal/utils/kill_windows.go`

> "On Windows — `kill_windows.go` — we use `CREATE_NEW_PROCESS_GROUP` and then `taskkill /T /F /PID` which recursively kills the process tree."

---

### 2g — Main Entry Point

**Open file:** `cmd/hotreload/main.go`

**What to say:**
> "The main.go wires everything together."

**Scroll to the component creation (line ~95-130):**
> "We create each component and connect them with channels: Watcher's `Events` channel feeds into the Debouncer, Debouncer's `Output` channel feeds into the Builder, and Builder's `Success` channel feeds into the Runner."

**Scroll to initial build trigger (line ~132-135):**
> "Critically, we trigger the first build immediately on startup — the developer doesn't need to edit a file to get the server running."

**Scroll to signal handling (line ~137-142):**
> "We handle SIGINT and SIGTERM for graceful shutdown. When you press Ctrl+C, everything shuts down cleanly."

---

## SECTION 3 — LIVE DEMO (2:30 – 6:30)

### What to say:
> "Now let me show all of this working live."

---

### Demo 1: Basic Hot Reload (2:30 – 3:30)

**Terminal 1 — Run this command:**
```powershell
.\bin\hotreload.exe --root .\samples\testserver --build "go build -o .\bin\server.exe ." --exec ".\bin\server.exe" --extensions ".go" --debounce 300
```

**What to say while it starts:**
> "I start hotreload pointing at our sample test server. Watch the logs — it immediately triggers the initial build without me saving anything."

**Point out the logs in Terminal 1:**
> "You can see: 'hotreload starting', 'watcher started with 1 directory', 'triggering initial build', 'build started', 'build succeeded' — and then 'server started' with a PID. The server is now running."

**Terminal 2 — Run:**
```powershell
curl.exe http://localhost:9090
```

**What to say:**
> "Let me curl it — 'hello world - pid=XXXX'. The server is running."

**Now open `samples/testserver/main.go` in VS Code.**

**What to say:**
> "Now I'll edit the server code. I'll change 'hello world' to 'hello hotreload'."

**Make the edit:** Change `"hello world - pid=%d\n"` to `"hello hotreload - pid=%d\n"` and **press Ctrl+S**.

**Point at Terminal 1 logs:**
> "Watch the logs — file change detected, debounce fires, build starts, old server stopped, new server started with a new PID. All automatic, under 2 seconds."

**Terminal 2 — Run:**
```powershell
curl.exe http://localhost:9090
```

**What to say:**
> "'hello hotreload' — and notice the PID changed. The old process was killed and a new one started with the updated code."

---

### Demo 2: Debounce — Rapid Saves (3:30 – 4:15)

**What to say:**
> "Now let me show debouncing. I'll rapidly save the file multiple times."

**In VS Code with `samples/testserver/main.go` open:**
- Add a space, Ctrl+S
- Add another space, Ctrl+S
- Delete space, Ctrl+S
- Delete space, Ctrl+S

(Do this quickly, 4 saves in about 1 second)

**Point at Terminal 1 logs:**
> "Look at the logs — you see multiple 'debounce: event received' entries, but only ONE 'debounce: emitting trigger'. All those rapid saves were coalesced into a single rebuild. Without this, the server would rebuild 4 times — with debouncing, just once."

---

### Demo 3: Build Failure — Syntax Error (4:15 – 5:15)

**What to say:**
> "What happens when you introduce a syntax error? The tool should handle it gracefully."

**In VS Code, open `samples/testserver/main.go`:**

**Make the edit:** Remove the closing parenthesis from the `fmt.Fprintf` line:
```go
// Change this:
fmt.Fprintf(w, "hello hotreload - pid=%d\n", os.Getpid())
// To this (delete the closing paren):
fmt.Fprintf(w, "hello hotreload - pid=%d\n", os.Getpid()
```

**Press Ctrl+S.**

**Point at Terminal 1 logs:**
> "Build started... and look — 'build failed', 'exit status 1', and the Go compiler error is right here: 'syntax error: unexpected newline'. The build failed, and importantly, the old server is still running."

**Terminal 2 — Run:**
```powershell
curl.exe http://localhost:9090
```

**What to say:**
> "See? The server still responds with the old code. The runner was never told to restart because the build failed. Your running server is safe even when you have a typo."

**Now fix the syntax error** — add back the closing paren, press Ctrl+S.

**Point at Terminal 1 logs:**
> "Now I fix it, save — build succeeds, old server killed, new server started. We're back online with the corrected code."

**Terminal 2 — Run:**
```powershell
curl.exe http://localhost:9090
```

> "Back to serving the correct response."

---

### Demo 4: Dynamic Directory Watching (5:15 – 5:45)

**What to say:**
> "The tool also handles projects that change structure while running."

**Terminal 2 — Run:**
```powershell
mkdir samples\testserver\newpkg
```

**Point at Terminal 1 logs:**
> "See the log — 'new directory detected: newpkg'. The watcher automatically starts monitoring it."

**Terminal 2 — Run:**
```powershell
Set-Content -Path samples\testserver\newpkg\helper.go -Value "package newpkg"
```

**Point at Terminal 1 logs:**
> "A new `.go` file in that directory triggers a rebuild automatically."

**Now delete the directory:**
```powershell
Remove-Item -Recurse -Force samples\testserver\newpkg
```

**Point at Terminal 1 logs:**
> "Directory deleted — the watcher cleans up the watch. No crash, no error spam. Graceful."

---

### Demo 5: Crash-Loop Detection (5:45 – 6:30)

**What to say:**
> "If the server crashes immediately after starting, the tool prevents an infinite restart loop."

**In VS Code, open `samples/testserver/main.go`:**

**Make the edit:** Add `os.Exit(1)` at the very start of `main()`:
```go
func main() {
	os.Exit(1)    // <-- add this line
```

**Press Ctrl+S.**

**Point at Terminal 1 logs:**
> "Watch — server starts, immediately exits, crash recorded — 1, 2, 3, 4, 5... and then: 'crash loop detected: server crashed too many times, pausing auto-restart'. The tool stopped restarting. It tells you to fix the code and save."

**Now remove the `os.Exit(1)` line and press Ctrl+S.**

**Point at Terminal 1 logs:**
> "I fix the code, save — the new file change resets the crash pause. Build succeeds, server starts and stays running. We're back to normal."

**Terminal 2 — Run:**
```powershell
curl.exe http://localhost:9090
```

> "Server is healthy again."

**Stop hotreload with Ctrl+C in Terminal 1.**

**What to say:**
> "And Ctrl+C gives us a clean graceful shutdown."

---

## SECTION 4 — REVERT TESTSERVER & TESTS (6:30 – 8:00)

### What to say:
> "Let me revert the test server to its original state and run the tests."

**In VS Code, revert `samples/testserver/main.go` back to:**
```go
fmt.Fprintf(w, "hello world - pid=%d\n", os.Getpid())
```
**(remove the "hotreload" change if still there)**

---

### 4a — Unit Tests

**Terminal — Run:**
```powershell
go test ./internal/... -v -race -count=1
```

**What to say while tests run:**
> "We have 33 unit tests across all packages, running with the Go race detector enabled."

**Point at the output as it flows:**

> "**Builder tests** — 4 tests: build success sends a signal to the runner, build failure does NOT send a signal, build cancellation when a new change arrives, and multiple successive builds work correctly."

> "**Debounce tests** — 6 tests: a single event produces one trigger, 50 rapid events get coalesced into one trigger, two bursts separated by more than the debounce window produce two triggers, zero events produce nothing, different files hitting at the same time are coalesced, and the timer resets on each new event."

> "**Runner tests** — 5 tests: start and stop a process, crash backoff detection after rapid failures, restart replaces the process with a new PID, a new build success resets the crash pause, and default config validation."

> "**Watcher tests** — 24 tests total: file create emits an event, file write emits an event, a new directory automatically gets watched, a deleted directory doesn't crash the watcher, ignored files produce no events, ignored directories aren't watched at all, nested directory creation works, watch count tracking, plus 16 ignore pattern tests covering defaults, custom patterns, extension whitelist, `.hotreloadignore` parsing, paths with spaces, deeply nested dirs, and case-insensitive matching."

**Point at the final output:**
> "All 33 tests pass. Zero race conditions detected."

---

### 4b — Integration Test

**Terminal — Run:**
```powershell
go test ./test/integration/... -v -race -timeout 60s -count=1
```

**What to say:**
> "Now the integration test — this is the real end-to-end test."

**Open file briefly:** `test/integration/integration_test.go`

> "It builds the hotreload binary from source, creates a temporary Go HTTP server project in a temp directory, starts hotreload against it, waits for the version-1 server to come up, then programmatically edits the source file to version-2, and waits for the server to restart and serve the new content. All automated. This is essentially what I just showed you in the live demo, but running automatically."

**Point at Terminal output:**
> "'server v1 is up, serving version-1' ... 'server v2 is up, serving version-2 — integration test passed!' — PASS. The full cycle works."

---

### 4c — Coverage

**Terminal — Run:**
```powershell
go test ./internal/... -race -coverprofile=coverage.out -covermode=atomic
go tool cover -func=coverage.out
```

**What to say:**
> "Coverage: builder 89.7%, debounce 87.5%, watcher 86.9%, runner 83.9%. All above 80%."

---

### 4d — CI

**What to show on screen:**
- **Browser → GitHub repo → Actions tab**

**What to say:**
> "CI runs on every push via GitHub Actions. The matrix tests on both Ubuntu and Windows. All green."

---

## SECTION 5 — CLOSING (8:00 – 8:30)

### What to say:
> "To summarize the key design decisions:"

> "First — **channels everywhere**. All components communicate through Go channels. No shared mutable state, no locks between components. Each component is a goroutine with clear input and output channels."

> "Second — **context.WithCancel** for build cancellation. If you save again while a build is running, the old build is cancelled immediately."

> "Third — **process-group kill**. We kill the entire process tree, not just the parent. This prevents zombie child processes and ensures the port is freed."

> "Fourth — **log/slog** for structured logging throughout. This is what Go recommends for production logging."

> "Fifth — **no framework dependencies**. The only external dependency is fsnotify for filesystem events. Everything else — debouncing, building, process management, crash detection — is implemented from scratch."

> "The tool is cross-platform — I developed on Windows but it runs on Linux and macOS too, with platform-specific process kill using Go build tags."

> "Thank you for watching!"

---

## QUICK REFERENCE — ALL COMMANDS USED IN VIDEO

| When | Command | Purpose |
|------|---------|---------|
| Demo setup | `go build -o bin/hotreload.exe ./cmd/hotreload` | Build the binary |
| Demo 1 | `.\bin\hotreload.exe --root .\samples\testserver --build "go build -o .\bin\server.exe ." --exec ".\bin\server.exe" --extensions ".go" --debounce 300` | Start hotreload |
| Demo 1 | `curl.exe http://localhost:9090` | Test server response |
| Demo 2 | Ctrl+S rapidly 4 times | Demonstrate debounce |
| Demo 3 | Delete closing paren → Ctrl+S | Introduce syntax error |
| Demo 3 | Fix paren → Ctrl+S | Fix syntax error |
| Demo 4 | `mkdir samples\testserver\newpkg` | Create new directory |
| Demo 4 | `Set-Content -Path samples\testserver\newpkg\helper.go -Value "package newpkg"` | Create file in new dir |
| Demo 4 | `Remove-Item -Recurse -Force samples\testserver\newpkg` | Delete directory |
| Demo 5 | Add `os.Exit(1)` to main → Ctrl+S | Trigger crash loop |
| Demo 5 | Remove `os.Exit(1)` → Ctrl+S | Fix crash loop |
| Tests | `go test ./internal/... -v -race -count=1` | Run 33 unit tests |
| Tests | `go test ./test/integration/... -v -race -timeout 60s -count=1` | Run integration test |
| Coverage | `go test ./internal/... -race -coverprofile=coverage.out -covermode=atomic` | Generate coverage |
| Coverage | `go tool cover -func=coverage.out` | Show coverage report |

---

## FILES REFERENCED IN VIDEO (in order)

| Section | File | What to highlight |
|---------|------|-------------------|
| Intro | `README.md` on GitHub | Features, architecture diagram, CI badge |
| Architecture | `internal/watcher/watcher.go` | `addRecursive()`, `handleEvent()`, `NewWithLimit()`, dynamic dir add/remove |
| Architecture | `internal/watcher/ignore.go` | `defaultIgnorePatterns`, `ShouldIgnore()`, `LoadHotreloadIgnore()` |
| Architecture | `internal/debounce/debounce.go` | `loop()` — timer reset logic |
| Architecture | `internal/builder/builder.go` | `runBuild()` — `context.WithCancel`, failure handling, success signal |
| Architecture | `internal/runner/runner.go` | `start()`, `stop()`, `recordCrash()` — process group, log streaming, crash backoff |
| Architecture | `internal/utils/kill_unix.go` | SIGTERM → SIGKILL to -PGID |
| Architecture | `internal/utils/kill_windows.go` | taskkill /T /F /PID |
| Architecture | `cmd/hotreload/main.go` | Channel wiring, initial build trigger, signal handling |
| Live Demo | `samples/testserver/main.go` | Edit greeting string, add/remove syntax error, add/remove os.Exit(1) |
| Tests | Terminal output | 33 unit tests + 1 integration test results |
| Tests | `test/integration/integration_test.go` | Briefly show the E2E test structure |
| CI | GitHub Actions tab in browser | Green matrix: ubuntu + windows |

---

## FEATURES CHECKLIST — CONFIRM ALL ARE SHOWN

| Feature | Where demonstrated |
|---------|--------------------|
| Recursive file watching | Demo 1 — server restarts on save |
| Event debouncing | Demo 2 — rapid saves produce single rebuild |
| Cancellable builds | Architecture section (builder.go) |
| Initial build on startup | Demo 1 — server starts without first edit |
| Real-time log streaming | Demo 1 — logs visible in terminal |
| Process-group kill | Architecture section (kill_unix.go, kill_windows.go) |
| Build failure handling | Demo 3 — syntax error, old server keeps running |
| Dynamic directory watching | Demo 4 — mkdir/rmdir while running |
| Crash-loop detection | Demo 5 — os.Exit(1) triggers backoff |
| .hotreloadignore support | Architecture section (ignore.go → LoadHotreloadIgnore) |
| Extension whitelist | Demo 1 — `--extensions ".go"` flag in command |
| Scalability guard | Architecture section (watcher.go → NewWithLimit) |
| Cross-platform | Architecture section (build tags), CI matrix |
| Graceful shutdown | Demo 5 — Ctrl+C |
| Structured logging (slog) | Visible in all demo terminal output |
| Tests with race detector | Section 4 — all 33 pass with -race |
| Integration test | Section 4 — end-to-end automated test |
| CI pipeline | Section 4 — GitHub Actions green |
