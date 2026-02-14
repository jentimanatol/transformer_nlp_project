git --version
git add .
git commit -m " update window title bar and taskbar icon to match the new logo and color scheme"
git push origin main

:: === Tagging for GitHub Actions Release Build ===
git tag v2.4
git push origin v2.4
pause
