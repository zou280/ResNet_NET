chcp 65001
@echo off
setlocal enabledelayedexpansion

rem 设置数据集文件夹路径
set "datasetFolder=F:\识别图像\训练自己的模型\bin\Debug\net8.0\训练数据集"

rem 创建或清空 images.tsv 文件
echo ImagePath Label > "%datasetFolder%\images.tsv"

rem 计算总文件数
set totalFiles=0
for /D %%d in ("%datasetFolder%\*") do (
    for %%f in ("%%~fd\*.*") do set /a totalFiles+=1
)

rem 初始化进度计数器
set currentFile=0

rem 遍历文件夹
for /D %%d in ("%datasetFolder%\*") do (
    rem 获取文件夹编号
    for %%i in (%%~nxd) do set label=%%i
    rem 遍历文件夹内的图片
    for %%f in ("%%~fd\*.*") do (
        rem 获取相对路径
        set "relativePath=%%~nxf"
        rem 写入 TSV 文件
        echo %%~fd\%%~nxf !label! >> "%datasetFolder%\images.tsv"
        rem 更新进度
        set /a currentFile+=1
        call :ShowProgress !currentFile! !totalFiles!
    )
)

echo.
echo images.tsv 文件已生成完成。
pause
exit /b

:ShowProgress
set /a percent=%1*100/%2
set /a bars=%percent%/2
set progressBar=
for /L %%i in (0,1,%bars%) do set progressBar=!progressBar!█
for /L %%i in (%bars%,1,50) do set progressBar=!progressBar!░
echo [!progressBar!] !percent!%%
exit /b