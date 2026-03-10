@echo off
setlocal enabledelayedexpansion

:: =============================================================
:: TabletScanner — Automated Build Script
:: Produces a fully packaged Electron + Python desktop app.
:: =============================================================

set "ROOT=%~dp0"
:: Remove trailing backslash for cleaner paths
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BACKEND=%ROOT%\backend"
set "FRONTEND=%ROOT%\frontend"
set "ELECTRON=%ROOT%\electron"
set "RELEASE=%ROOT%\release"

echo.
echo ============================================================
echo  TabletScanner Build Script
echo ============================================================
echo  Root:      %ROOT%
echo  Backend:   %BACKEND%
echo  Frontend:  %FRONTEND%
echo  Electron:  %ELECTRON%
echo  Output:    %RELEASE%
echo ============================================================
echo.

:: -----------------------------------------------------------
:: Ask: install dependencies?
:: -----------------------------------------------------------
set "INSTALL_DEPS=N"
set /p "INSTALL_DEPS=Install/update dependencies? (pip, npm) [y/N]: "
if /i "!INSTALL_DEPS!"=="" set "INSTALL_DEPS=N"
echo.

:: -----------------------------------------------------------
:: Prepare app icon for Electron (must include 256x256)
:: -----------------------------------------------------------
set "ICON_SOURCE_PNG=%FRONTEND%\public\app-icon.png"
if not exist "%ICON_SOURCE_PNG%" set "ICON_SOURCE_PNG=%FRONTEND%\src\assets\logo.png"
set "ICON_SOURCE_ICO=%FRONTEND%\public\favicon.ico"

if exist "%ICON_SOURCE_PNG%" (
    echo Preparing icon from PNG source: %ICON_SOURCE_PNG%
    python -c "from PIL import Image; s=r'%ICON_SOURCE_PNG%'; d=r'%ELECTRON%\icon.ico'; im=Image.open(s).convert('RGBA'); w,h=im.size; side=max(w,h,512); bg=Image.new('RGBA',(side,side),(0,0,0,0)); bg.paste(im,((side-w)//2,(side-h)//2),im if im.mode=='RGBA' else None); sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)]; bg.save(d, format='ICO', sizes=sizes); print('Icon written:', d, 'base:', bg.size)"
    if !errorlevel! neq 0 (
        echo ERROR: Failed to generate icon.ico from PNG source.
        exit /b 1
    )
) else if exist "%ICON_SOURCE_ICO%" (
    echo Preparing icon from ICO source: %ICON_SOURCE_ICO%
    python -c "from PIL import Image; s=r'%ICON_SOURCE_ICO%'; d=r'%ELECTRON%\icon.ico'; im=Image.open(s).convert('RGBA'); w,h=im.size; side=max(w,h,512); bg=Image.new('RGBA',(side,side),(0,0,0,0)); bg.paste(im,((side-w)//2,(side-h)//2),im if im.mode=='RGBA' else None); sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)]; bg.save(d, format='ICO', sizes=sizes); print('Icon written:', d, 'base:', bg.size)"
    if !errorlevel! neq 0 (
        echo ERROR: Failed to normalize favicon.ico into icon.ico.
        exit /b 1
    )
) else (
    echo ERROR: No icon source found.
    echo Place an icon source at one of these paths:
    echo   - %FRONTEND%\public\app-icon.png  ^(preferred: 512x512^)
    echo   - %FRONTEND%\src\assets\logo.png
    echo   - %FRONTEND%\public\favicon.ico
    exit /b 1
)

:: -----------------------------------------------------------
:: Step 1: Build Python backend with PyInstaller
:: -----------------------------------------------------------
echo [1/6] Building Python backend with PyInstaller ...
pushd "%BACKEND%"

if /i "!INSTALL_DEPS!"=="y" (
    echo        [1a] Installing Python dependencies ...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install Python dependencies.
        popd & exit /b 1
    )
    echo        [1a] Python dependencies installed.

    echo        [1b] Installing PyInstaller ...
    pip install pyinstaller
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install PyInstaller.
        popd & exit /b 1
    )
    echo        [1b] PyInstaller installed.
) else (
    echo        Skipping dependency install (using existing packages^)
)

echo        [1c] Running PyInstaller (this may take a few minutes) ...
python -m PyInstaller app.py ^
    --noconsole ^
    --add-data "error_messages.json;." ^
    --add-data "settings.json;." ^
    --distpath dist ^
    --workpath build ^
    --specpath . ^
    -y
if !errorlevel! neq 0 (
    echo ERROR: PyInstaller build failed.
    popd & exit /b 1
)
echo        [1c] PyInstaller packaging complete.

popd
echo [1/6] Backend build complete.
echo.

:: -----------------------------------------------------------
:: Step 2: Build Angular frontend (production)
:: -----------------------------------------------------------
echo [2/6] Building Angular frontend ...
pushd "%FRONTEND%"

if /i "!INSTALL_DEPS!"=="y" (
    echo        Installing npm dependencies ...
    call npm install
    if !errorlevel! neq 0 (
        echo ERROR: npm install failed.
        popd & exit /b 1
    )
) else (
    echo        Skipping npm install (using existing node_modules^)
)

call npx ng build --configuration production --base-href ./
if !errorlevel! neq 0 (
    echo ERROR: Angular build failed.
    popd & exit /b 1
)

popd
echo [2/6] Frontend build complete.
echo.

:: -----------------------------------------------------------
:: Step 3: Copy Angular output into Electron app folder
:: -----------------------------------------------------------
echo [3/6] Copying frontend build to Electron app folder ...

:: Clean previous app folder
if exist "%ELECTRON%\app" rmdir /s /q "%ELECTRON%\app"
mkdir "%ELECTRON%\app"

:: Angular 20 application builder outputs to dist/<project>/browser
set "NG_OUTPUT=%FRONTEND%\dist\tabletscanner\browser"
if not exist "%NG_OUTPUT%" (
    echo ERROR: Angular output not found at %NG_OUTPUT%
    exit /b 1
)

xcopy "%NG_OUTPUT%\*" "%ELECTRON%\app\" /s /e /q /y
echo [3/6] Frontend copied.
echo.

:: -----------------------------------------------------------
:: Step 4: Install Electron dependencies
:: -----------------------------------------------------------
echo [4/6] Installing Electron dependencies ...
pushd "%ELECTRON%"

if /i "!INSTALL_DEPS!"=="y" (
    echo        Installing Electron npm dependencies ...
    call npm install
    if !errorlevel! neq 0 (
        echo ERROR: Electron npm install failed.
        popd & exit /b 1
    )
) else (
    if not exist "%ELECTRON%\node_modules" (
        echo        node_modules missing — running npm install anyway ...
        call npm install
        if !errorlevel! neq 0 (
            echo ERROR: Electron npm install failed.
            popd & exit /b 1
        )
    ) else (
        echo        Skipping npm install (using existing node_modules^)
    )
)

popd
echo [4/6] Electron dependencies installed.
echo.

:: -----------------------------------------------------------
:: Step 5: Build Electron application
:: -----------------------------------------------------------
echo [5/6] Building Electron application ...
pushd "%ELECTRON%"

:: Clean previous release output
if exist "%RELEASE%" rmdir /s /q "%RELEASE%"

call npx electron-builder --win
if !errorlevel! neq 0 (
    echo ERROR: Electron build failed.
    popd & exit /b 1
)

popd
echo [5/6] Electron build complete.
echo.

:: -----------------------------------------------------------
:: Step 6: Copy backend into Electron package resources
:: -----------------------------------------------------------
echo [6/6] Copying backend into packaged application ...

set "RESOURCES=%RELEASE%\win-unpacked\resources\backend"
mkdir "%RESOURCES%" 2>nul

:: Copy entire PyInstaller output
xcopy "%BACKEND%\dist\app\*" "%RESOURCES%\" /s /e /q /y

:: Ensure settings.json is in the backend resource folder
copy /y "%BACKEND%\settings.json" "%RESOURCES%\settings.json" >nul

echo [6/6] Backend copied to resources.
echo.

:: -----------------------------------------------------------
:: Done
:: -----------------------------------------------------------
echo ============================================================
echo  BUILD COMPLETE
echo.
echo  Output: %RELEASE%\win-unpacked\
echo.
echo  To run: %RELEASE%\win-unpacked\TabletScanner.exe
echo ============================================================

endlocal
