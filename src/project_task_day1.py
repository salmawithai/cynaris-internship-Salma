# Week 1, Day 1 — Environment Verification Script
import sys
import subprocess

print('=== Python Environment Check ===')
print(f'Python version: {sys.version}')
print(f'Python path: {sys.executable}')

tools = ['git', 'docker']
for tool in tools:
    try:
        result = subprocess.run([tool, '--version'], capture_output=True, text=True)
        print(f'{tool}: {result.stdout.strip()[:50]}')
    except FileNotFoundError:
        print(f'{tool}: NOT FOUND — install before proceeding')

print('\n✅ If all checks pass, proceed to install your project')
print('📋 Note: CrewAI multi-agent framework')