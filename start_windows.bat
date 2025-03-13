@echo off
setlocal enabledelayedexpansion

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

echo "%CD%"| findstr /C:" " >nul && echo �˽ű�����Miniconda�������޷��ڰ����ո��·���¾�Ĭ��װ�� && goto end

@rem Check for special characters in installation path
set "SPCHARMESSAGE="���棺�ڰ�װ·���м�⵽�����ַ���" "         ����ܵ��°�װʧ�ܣ�""
echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~]" >nul && (
	call :PrintBigMessage %SPCHARMESSAGE%
)
set SPCHARMESSAGE=

@rem fix failed install when installing to a separate drive
set TMP=%cd%\installer_files
set TEMP=%cd%\installer_files

@rem deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

@rem config
set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MINICONDA_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_24.11.1-0-Windows-x86_64.exe
set MINICONDA_CHECKSUM=43dcbcc315ff91edf959e002cd2f1ede38c64b999fefcc951bccf2ed69c9e8bb
set conda_exists=F

@rem figure out whether git and conda needs to be installed
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

@rem (if necessary) install git and conda into a contained environment
@rem download conda
if "%conda_exists%" == "F" (
	echo ���ڴ� %MINICONDA_DOWNLOAD_URL% ����Miniconda�� %INSTALL_DIR%\miniconda_installer.exe

	mkdir "%INSTALL_DIR%"
	call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe" || ( echo. && echo ����Minicondaʧ�ܡ� && goto end )

	@rem Try CertUtil first
	for /f %%a in ('CertUtil -hashfile "%INSTALL_DIR%\miniconda_installer.exe" SHA256 ^| find /i /v " " ^| find /i "%MINICONDA_CHECKSUM%"') do (
		set "output=%%a"
	)

	@rem If CertUtil fails, try PowerShell
	if not defined output (
		for /f %%a in ('powershell -Command "if((Get-FileHash \"%INSTALL_DIR%\miniconda_installer.exe\" -Algorithm SHA256).Hash -eq ''%MINICONDA_CHECKSUM%''){echo true}"') do (
			set "output=%%a"
		)
	)

	if not defined output (
		echo miniconda_installer.exe��У�����֤ʧ���ˡ�
		del "%INSTALL_DIR%\miniconda_installer.exe"
		goto end
	) else (
		echo miniconda_installer.exe��У�����֤�Ѿ��ɹ�ͨ����
	)

	echo ���ڽ�Miniconda��װ�� %CONDA_ROOT_PREFIX%
	start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

	@rem test the conda binary
	echo Miniconda�汾��
	call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || ( echo. && echo �Ҳ���Miniconda�� && goto end )

	@rem delete the Miniconda installer
	del "%INSTALL_DIR%\miniconda_installer.exe"
)

@rem create the installer env
if not exist "%INSTALL_ENV_DIR%" (
	echo ������װ���������� %PACKAGES_TO_INSTALL%
	call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.11 || ( echo. && echo ����Conda����ʧ�ܡ� && goto end )
)

@rem check if conda environment was actually created
if not exist "%INSTALL_ENV_DIR%\python.exe" ( echo. && echo Conda����δ������ && goto end )

@rem environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo �Ҳ���Miniconda���ӡ� && goto end )

@rem setup installer env
call python one_click.py %*

@rem below are functions for the script   next line skips these during normal execution
goto end

:PrintBigMessage
echo. && echo.
echo *******************************************************************
for %%M in (%*) do echo * %%~M
echo *******************************************************************
echo. && echo.
exit /b

:end
pause
