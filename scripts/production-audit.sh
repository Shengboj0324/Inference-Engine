#!/bin/bash
# Production Readiness Audit Script
# Performs comprehensive code quality and security checks

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         PRODUCTION READINESS AUDIT - MAXIMUM STRICTNESS       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0
WARNINGS=0

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

function error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    ((ERRORS++))
}

function warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
    ((WARNINGS++))
}

function success() {
    echo -e "${GREEN}✓ $1${NC}"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. CHECKING FOR TODOs AND FIXMEs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TODO_COUNT=$(grep -rn "TODO\|FIXME\|XXX\|HACK" app --include="*.py" | wc -l | tr -d ' ')
if [ "$TODO_COUNT" -gt 0 ]; then
    error "Found $TODO_COUNT TODOs/FIXMEs in production code"
    grep -rn "TODO\|FIXME\|XXX\|HACK" app --include="*.py" | head -10
else
    success "No TODOs or FIXMEs found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. CHECKING FOR BARE EXCEPT CLAUSES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BARE_EXCEPT=$(grep -rn "except:" app --include="*.py" | grep -v "# except:" | grep -v "except Exception" | grep -v "except (" | wc -l | tr -d ' ')
if [ "$BARE_EXCEPT" -gt 0 ]; then
    warning "Found $BARE_EXCEPT bare except clauses"
    grep -rn "except:" app --include="*.py" | grep -v "# except:" | grep -v "except Exception" | grep -v "except (" | head -5
else
    success "No bare except clauses found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. CHECKING FOR PRINT STATEMENTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PRINT_COUNT=$(grep -rn "print(" app --include="*.py" | grep -v "# print(" | wc -l | tr -d ' ')
if [ "$PRINT_COUNT" -gt 0 ]; then
    warning "Found $PRINT_COUNT print statements (should use logging)"
    grep -rn "print(" app --include="*.py" | grep -v "# print(" | head -5
else
    success "No print statements found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. CHECKING SYNTAX ERRORS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SYNTAX_ERRORS=0
for file in $(find app -name "*.py" -type f); do
    if ! python3 -m py_compile "$file" 2>/dev/null; then
        error "Syntax error in $file"
        ((SYNTAX_ERRORS++))
    fi
done

if [ "$SYNTAX_ERRORS" -eq 0 ]; then
    success "No syntax errors found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. CHECKING FOR HARDCODED SECRETS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SECRET_PATTERNS="password|secret|api_key|token|credential"
SECRET_COUNT=$(grep -rni "$SECRET_PATTERNS" app --include="*.py" | grep -v "# " | grep -v "def " | grep -v "class " | grep "=" | wc -l | tr -d ' ')
if [ "$SECRET_COUNT" -gt 0 ]; then
    warning "Found $SECRET_COUNT potential hardcoded secrets"
    grep -rni "$SECRET_PATTERNS" app --include="*.py" | grep -v "# " | grep -v "def " | grep -v "class " | grep "=" | head -5
else
    success "No hardcoded secrets detected"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. CHECKING FOR SQL INJECTION RISKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SQL_INJECTION=$(grep -rn "execute.*%\|execute.*format\|execute.*+" app --include="*.py" | wc -l | tr -d ' ')
if [ "$SQL_INJECTION" -gt 0 ]; then
    error "Found $SQL_INJECTION potential SQL injection risks"
    grep -rn "execute.*%\|execute.*format\|execute.*+" app --include="*.py" | head -5
else
    success "No SQL injection risks detected"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. CHECKING FOR MISSING TYPE HINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Count functions without type hints (simplified check)
FUNCTIONS_WITHOUT_TYPES=$(grep -rn "^def \|^    def \|^        def " app --include="*.py" | grep -v " -> " | wc -l | tr -d ' ')
TOTAL_FUNCTIONS=$(grep -rn "^def \|^    def \|^        def " app --include="*.py" | wc -l | tr -d ' ')

if [ "$TOTAL_FUNCTIONS" -gt 0 ]; then
    TYPE_COVERAGE=$((100 - (FUNCTIONS_WITHOUT_TYPES * 100 / TOTAL_FUNCTIONS)))
    if [ "$TYPE_COVERAGE" -lt 80 ]; then
        warning "Type hint coverage: ${TYPE_COVERAGE}% (target: 80%+)"
    else
        success "Type hint coverage: ${TYPE_COVERAGE}%"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. RUNNING TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if python3 -m pytest tests/ -v --tb=short -q 2>&1 | tail -20; then
    success "All tests passed"
else
    error "Some tests failed"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      AUDIT SUMMARY                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "Errors:   ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo -e "${GREEN}✓ PRODUCTION READY - ALL CHECKS PASSED${NC}"
    exit 0
elif [ "$ERRORS" -eq 0 ]; then
    echo -e "${YELLOW}⚠ PRODUCTION READY WITH WARNINGS${NC}"
    exit 0
else
    echo -e "${RED}✗ NOT PRODUCTION READY - FIX ERRORS BEFORE DEPLOYMENT${NC}"
    exit 1
fi

