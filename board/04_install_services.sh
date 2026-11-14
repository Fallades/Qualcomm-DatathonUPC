#!/usr/bin/env bash
# Run both llama-servers as systemd services, so they start at boot and restart
# if they crash. They listen on 127.0.0.1 only: nothing is exposed to the network.
# Run it again whenever you change config.env.
source "$(dirname "$0")/common.sh"

[ -x "$LLAMA_SERVER" ] || die "llama-server not found. Run ./board/02_build_llama_cpp.sh first."
[ -f "$LLM_PATH" ] && [ -f "$EMBED_PATH" ] && [ -f "$VISION_MMPROJ_PATH" ] ||
  die "models not found. Run ./board/03_download_models.sh first."

# A llama-server you started by hand would hold the port and stop the service
for port in "$LLM_PORT" "$EMBED_PORT" "$VISION_PORT"; do
  pid="$(pgrep -f -- "llama-server.*--port $port" | head -1 || true)"
  service_pids="$(systemctl show -p MainPID --value math-llm math-embed math-vision 2>/dev/null || true)"
  if [ -n "$pid" ] && [[ $'\n'"$service_pids"$'\n' != *$'\n'"$pid"$'\n'* ]]; then
    die "a llama-server you started by hand (PID $pid) is using port $port. Stop it with: kill $pid"
  fi
done

render() {
  sed -e "s|@USER@|$USER|g" \
      -e "s|@LLAMA_SERVER@|$LLAMA_SERVER|g" \
      -e "s|@LLM_PATH@|$LLM_PATH|g" -e "s|@LLM_PORT@|$LLM_PORT|g" \
      -e "s|@LLM_THREADS@|$LLM_THREADS|g" -e "s|@LLM_CTX@|$LLM_CTX|g" -e "s|@LLM_ALIAS@|$LLM_ALIAS|g" \
      -e "s|@EMBED_PATH@|$EMBED_PATH|g" -e "s|@EMBED_PORT@|$EMBED_PORT|g" \
      -e "s|@EMBED_THREADS@|$EMBED_THREADS|g" -e "s|@EMBED_CTX@|$EMBED_CTX|g" \
      -e "s|@VISION_MMPROJ_PATH@|$VISION_MMPROJ_PATH|g" \
      -e "s|@VISION_PORT@|$VISION_PORT|g" \
      -e "s|@VISION_IMAGE_MAX_TOKENS@|$VISION_IMAGE_MAX_TOKENS|g" \
      "$REPO_DIR/board/systemd/$1"
}

step "Installing math-llm.service and math-embed.service"
for unit in math-llm.service math-embed.service math-vision.service; do
  render "$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable math-llm math-embed math-vision
sudo systemctl restart math-llm math-embed math-vision

step "Waiting for the servers to load the models"
wait_for_health "$LLM_PORT" 180 || die "the LLM server didn't start. See: journalctl -u math-llm -n 50"
ok "LLM server ready on 127.0.0.1:$LLM_PORT"
wait_for_health "$EMBED_PORT" 60 || die "the embedding server didn't start. See: journalctl -u math-embed -n 50"
ok "embedding server ready on 127.0.0.1:$EMBED_PORT"
wait_for_health "$VISION_PORT" 180 || die "the vision server didn't start. See: journalctl -u math-vision -n 50"
ok "vision server ready on 127.0.0.1:$VISION_PORT"
free -m | awk 'NR==2 {printf "  RAM: %d MB used, %d MB available\n", $3, $7}'
ok "Next: ./board/05_smoke_test.sh"
