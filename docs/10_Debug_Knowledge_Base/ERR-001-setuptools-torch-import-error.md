## ERR-001: setuptools版本导致torch模块导入失败

### Date

2026-03-02

### Environment

- OS: Linux
- Component: setup.py编译
- Python Package: setuptools

### Symptom

在执行`python setup.py`编译时,持续报错:

```
ModuleNotFoundError: No module named 'torch'
```

尽管环境中已正确安装torch模块,但编译过程仍无法识别。

### Debug Process

1. 确认环境中torch模块已正确安装,可以正常导入
2. 检查setup.py的依赖配置,未发现异常
3. 搜索相关错误,发现是setuptools版本兼容性问题
4. 参考链接: https://blog.csdn.net/weixin_43912083/article/details/148203228

### Root Cause

setuptools版本过高(80.8)导致在setup.py编译过程中无法正确识别已安装的torch模块。这是setuptools在新版本中的行为变更导致的兼容性问题。

### Fix

1. 降低setuptools版本从80.8到78.1.1:
   
   ```bash
   pip install setuptools==78.1.1
   ```
2. 重新执行setup.py编译

### Verification

执行setup.py编译成功,不再报"no module named torch"错误。

### Lessons & Notes

- setuptools版本过高可能导致依赖包识别问题
- 遇到"模块未找到"错误时,除了检查模块是否安装,还应考虑构建工具的版本兼容性
- 建议在requirements.txt或setup.py中明确指定setuptools版本范围,避免自动升级导致的问题
