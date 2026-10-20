@echo off
cd /d "%~dp0"
echo =======================================================
echo Pushing Trace-Ai to GitHub (account: sharuk-ahamed-y)
echo =======================================================
echo.
echo GitHub no longer supports standard passwords for git push.
echo Please provide a Personal Access Token (PAT) or press Enter to login via browser.
echo.
set /p TOKEN="Enter GitHub Personal Access Token (or press Enter for browser login): "

if "%TOKEN%"=="" (
    echo.
    echo Removing cached credentials from Windows Credential Manager...
    cmdkey /delete:LegacyGeneric:target=git:https://github.com >nul 2>&1
    git remote set-url origin https://github.com/sharuk-ahamed-y/Trace-Ai.git
    echo.
    echo Opening GitHub browser sign-in...
    git push -u origin main
) else (
    echo.
    echo Pushing to repository using Personal Access Token...
    git push https://sharuk-ahamed-y:%TOKEN%@github.com/sharuk-ahamed-y/Trace-Ai.git main
    git remote set-url origin https://github.com/sharuk-ahamed-y/Trace-Ai.git
)

echo.
pause
