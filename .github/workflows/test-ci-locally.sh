#!/bin/bash
# Script to test CI workflow locally before pushing

set -e

echo "=== Testing CI Workflow Locally ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "1. Checking Python version..."
python3 --version || { echo -e "${RED}❌ Python 3 not found${NC}"; exit 1; }
echo -e "${GREEN}✅ Python OK${NC}"
echo ""

# Check if pyproject.toml exists
echo "2. Checking pyproject.toml..."
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ pyproject.toml not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ pyproject.toml found${NC}"
echo ""

# Test syntax of all Python files
echo "3. Testing Python syntax..."
find app tests -name "*.py" -type f -exec python3 -m py_compile {} \; 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All Python files have valid syntax${NC}"
else
    echo -e "${RED}❌ Syntax errors found${NC}"
    exit 1
fi
echo ""

# Test imports
echo "4. Testing critical imports..."
python3 -c "
import sys
sys.path.insert(0, '.')

test_modules = [
    'app.core.models',
    'app.core.config',
]

failed = []
for module in test_modules:
    try:
        __import__(module)
        print(f'  ✅ {module}')
    except Exception as e:
        print(f'  ❌ {module}: {e}')
        failed.append(module)

if failed:
    print(f'\n❌ {len(failed)} modules failed')
    sys.exit(1)
else:
    print(f'\n✅ All critical modules import successfully')
" || { echo -e "${RED}❌ Import test failed${NC}"; exit 1; }
echo ""

# Check for common issues
echo "5. Checking for common issues..."

# Bare except clauses
echo "  - Checking for bare except clauses..."
BARE_EXCEPT=$(grep -rn "except:" app --include="*.py" | grep -v "# except:" | wc -l)
if [ "$BARE_EXCEPT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $BARE_EXCEPT bare except clauses${NC}"
else
    echo -e "${GREEN}  ✅ No bare except clauses${NC}"
fi

# Print statements
echo "  - Checking for print statements..."
PRINTS=$(grep -rn "print(" app --include="*.py" | grep -v "# print(" | wc -l)
if [ "$PRINTS" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $PRINTS print statements${NC}"
else
    echo -e "${GREEN}  ✅ No print statements${NC}"
fi

# TODO comments
echo "  - Checking for TODO comments..."
TODOS=$(grep -rn "TODO" app --include="*.py" | wc -l)
if [ "$TODOS" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $TODOS TODO comments${NC}"
else
    echo -e "${GREEN}  ✅ No TODO comments${NC}"
fi
echo ""

# Check CI workflow syntax
echo "6. Checking GitHub Actions workflow syntax..."
if command -v yamllint &> /dev/null; then
    yamllint .github/workflows/ci.yml || echo -e "${YELLOW}⚠️  YAML lint warnings${NC}"
    echo -e "${GREEN}✅ Workflow YAML checked${NC}"
else
    echo -e "${YELLOW}⚠️  yamllint not installed, skipping${NC}"
fi
echo ""

# Summary
echo "=== Summary ==="
echo -e "${GREEN}✅ All basic checks passed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Commit your changes: git add . && git commit -m 'Fix CI issues'"
echo "  2. Push to GitHub: git push"
echo "  3. Check GitHub Actions: https://github.com/YOUR_USERNAME/YOUR_REPO/actions"
echo ""
echo "Note: Some checks (like dependency installation) can only be tested in CI"

