"""Verify release declarations and run available regression tests."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
actual = set()
for path in (root/'mednet').rglob('*.py'):
    source = path.read_text(encoding='utf-8')
    compile(source, str(path), 'exec')
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            nodes.extend((node.name+'.'+m.name, m) for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)))
    for symbol, node in nodes:
        body = list(node.body)
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if len(body) == 1 and isinstance(body[0], ast.Raise) and isinstance(body[0].exc, ast.Call) and isinstance(body[0].exc.func, ast.Name) and body[0].exc.func.id == 'NotImplementedError':
            actual.add((path.relative_to(root).as_posix(), symbol))
            try:
                exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])), 'gate', 'exec'), {})
            except NotImplementedError:
                pass
            else:
                raise RuntimeError('Release gate did not raise')
expected = {(r['file'],r['symbol']) for r in manifest['scheduled']}
if len(expected) != len(manifest['scheduled']) or actual != expected:
    raise RuntimeError('Release manifest does not match source gates')
print(manifest['version'], 'syntax and release declarations verified', flush=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=root, env=env, check=True)
print('Available tests passed. Synthetic checks do not establish detection accuracy.')
