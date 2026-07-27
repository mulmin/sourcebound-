# 테스터 피드백 확인 — sourcebound
# 무료/서버리스 환경에선 파일이 재시작마다 날아가므로, 피드백을 로그(stdout)로도 남겨둠.
# 이 스크립트는 Fly 로그에서 [FEEDBACK] 줄만 뽑아 보여준다.
#
# 사용법:  powershell -ExecutionPolicy Bypass -File ops\check_feedback.ps1
#          (수집된 최근 로그를 훑음. 실시간으로 계속 보려면 아래 -Follow 옵션)

param(
    [switch]$Follow  # 실시간 스트리밍으로 계속 지켜보기
)

$env:PATH += ";C:\Users\mulmin\.fly\bin"

Write-Host "=== sourcebound 피드백 (로그의 [FEEDBACK]) ===" -ForegroundColor Cyan
Write-Host "(👍/👎 와 코멘트가 남은 줄만 표시)`n" -ForegroundColor DarkGray

if ($Follow) {
    Write-Host "실시간 감시 중… (Ctrl+C로 종료)`n" -ForegroundColor Yellow
    fly logs -a sourcebound | Select-String -Pattern "\[FEEDBACK\]"
} else {
    # 최근 로그 스냅샷에서 피드백만 추림
    $lines = fly logs -a sourcebound --no-tail 2>&1 | Select-String -Pattern "\[FEEDBACK\]"
    if (-not $lines) {
        Write-Host "아직 수집된 피드백이 없습니다. (또는 로그 보존 기간이 지났을 수 있음)" -ForegroundColor DarkGray
        Write-Host "실시간으로 지켜보려면:  ops\check_feedback.ps1 -Follow" -ForegroundColor DarkGray
    } else {
        $lines | ForEach-Object { Write-Host $_.Line }
        Write-Host "`n총 $($lines.Count) 건" -ForegroundColor Green
    }
}
