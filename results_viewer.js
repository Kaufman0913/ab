#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ANSI escape codes for styling
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    white: '\x1b[37m',
    bgBlack: '\x1b[40m',
    bgWhite: '\x1b[47m',
    clearScreen: '\x1b[2J',
    moveCursor: '\x1b[H'
};

class ResultsViewer {
    constructor(runsDirectory) {
        this.runsDirectory = runsDirectory;
        this.results = [];
        this.currentIndex = 0;
        this.selectedResult = null;
        this.viewMode = 'list'; // 'list' or 'detail'
        
        this.loadResults();
        this.setupInput();
    }

    loadResults() {
        try {
            const files = fs.readdirSync(this.runsDirectory)
                .filter(file => file.endsWith('.json'))
                .sort();

            this.results = files.map(file => {
                const filePath = path.join(this.runsDirectory, file);
                try {
                    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                    const status = this.determineStatus(data);
                    return {
                        filename: file,
                        name: file.replace('.json', ''),
                        data: data,
                        status: status,
                        emoji: status === 'pass' ? '✅' : '❌'
                    };
                } catch (error) {
                    return {
                        filename: file,
                        name: file.replace('.json', ''),
                        data: null,
                        status: 'error',
                        emoji: '❌',
                        error: error.message
                    };
                }
            });

            console.log(`${colors.green}Loaded ${this.results.length} results from ${this.runsDirectory}${colors.reset}`);
        } catch (error) {
            console.error(`${colors.red}Error loading results: ${error.message}${colors.reset}`);
            process.exit(1);
        }
    }

    determineStatus(data) {
        // Check if both agent_result and eval_result exist and have status "success"
        if (!data.agent_result || data.agent_result.status !== 'success') {
            return 'fail';
        }

        // If eval_result doesn't exist, it's a fail
        if (!data.eval_result || data.eval_result.status !== 'success') {
            return 'fail';
        }

        // Check if all tests in eval_result passed
        if (data.eval_result.test_results) {
            const allTestsPassed = data.eval_result.test_results.every(test => test.status === 'pass');
            return allTestsPassed ? 'pass' : 'fail';
        }

        // If we have eval_result with success status but no test_results, consider it a pass
        return 'pass';
    }

    setupInput() {
        readline.emitKeypressEvents(process.stdin);
        if (process.stdin.isTTY) {
            process.stdin.setRawMode(true);
        }

        process.stdin.on('keypress', (str, key) => {
            if (key.ctrl && key.name === 'c') {
                this.cleanup();
                process.exit(0);
            }

            this.handleKeypress(key);
        });
    }

    handleKeypress(key) {
        if (this.viewMode === 'list') {
            this.handleListKeypress(key);
        } else {
            this.handleDetailKeypress(key);
        }
        this.render();
    }

    handleListKeypress(key) {
        const terminalWidth = process.stdout.columns || 80;
        const maxNameLength = Math.max(...this.results.map(r => r.name.length));
        const columnWidth = Math.max(maxNameLength + 4, 20);
        const columnsPerRow = Math.floor(terminalWidth / columnWidth);

        switch (key.name) {
            case 'up':
                this.currentIndex = Math.max(0, this.currentIndex - columnsPerRow);
                break;
            case 'down':
                this.currentIndex = Math.min(this.results.length - 1, this.currentIndex + columnsPerRow);
                break;
            case 'left':
                this.currentIndex = Math.max(0, this.currentIndex - 1);
                break;
            case 'right':
                this.currentIndex = Math.min(this.results.length - 1, this.currentIndex + 1);
                break;
            case 'return':
            case 'space':
                this.selectedResult = this.results[this.currentIndex];
                this.viewMode = 'detail';
                break;
            case 'q':
                this.cleanup();
                process.exit(0);
                break;
        }
    }

    handleDetailKeypress(key) {
        switch (key.name) {
            case 'escape':
            case 'backspace':
            case 'q':
                this.viewMode = 'list';
                this.selectedResult = null;
                break;
        }
    }

    render() {
        console.log(colors.clearScreen + colors.moveCursor);

        if (this.viewMode === 'list') {
            this.renderList();
        } else {
            this.renderDetail();
        }
    }

    renderList() {
        console.log(`${colors.bright}${colors.cyan}🔍 Test Results Viewer${colors.reset}\n`);
        
        const passCount = this.results.filter(r => r.status === 'pass').length;
        const failCount = this.results.filter(r => r.status === 'fail').length;
        const errorCount = this.results.filter(r => r.status === 'error').length;

        console.log(`${colors.green}✅ Pass: ${passCount}${colors.reset} | ${colors.red}❌ Fail: ${failCount}${colors.reset} | ${colors.yellow}⚠️  Error: ${errorCount}${colors.reset}\n`);

        console.log(`${colors.dim}Use ↑/↓/←/→ arrows to navigate, Enter/Space to view details, 'q' to quit${colors.reset}\n`);

        this.renderGrid();
    }

    renderGrid() {
        const terminalWidth = process.stdout.columns || 80;
        const maxNameLength = Math.max(...this.results.map(r => r.name ? r.name.length : 0));
        const columnWidth = Math.max(maxNameLength + 4, 20); // +4 for emoji and padding
        const columnsPerRow = Math.floor(terminalWidth / columnWidth);
        
        for (let i = 0; i < this.results.length; i += columnsPerRow) {
            const rowResults = this.results.slice(i, i + columnsPerRow);
            let line = '';
            
            rowResults.forEach((result, colIndex) => {
                const globalIndex = i + colIndex;
                const isSelected = globalIndex === this.currentIndex;
                
                // Debug: Check if result is valid
                if (!result) {
                    console.error('Invalid result at index:', globalIndex);
                    return;
                }
                
                // Ensure result has required properties with fallbacks
                const emoji = result.emoji || '❓';
                const name = result.name || 'unknown';
                const status = result.status || 'unknown';
                
                // Build item string
                let item = emoji + ' ' + name;
                
                // Truncate if too long for column
                if (item.length > columnWidth - 1) {
                    item = item.substring(0, columnWidth - 4) + '...';
                }
                
                // Pad to column width BEFORE adding colors
                item = item.padEnd(columnWidth - 1);
                
                // Apply colors after padding
                if (isSelected) {
                    item = colors.bgWhite + '\x1b[30m' + item + colors.reset;
                } else if (status === 'pass') {
                    item = colors.green + item + colors.reset;
                } else if (status === 'error') {
                    item = colors.yellow + item + colors.reset;
                } else {
                    item = colors.red + item + colors.reset;
                }
                
                line += item + ' ';
            });
            
            console.log(line);
        }
    }

    renderDetail() {
        const result = this.selectedResult;
        console.log(`${colors.bright}${colors.cyan}📋 ${result.name}${colors.reset}\n`);
        
        console.log(`${colors.dim}Press 'q', Esc, or Backspace to return to list${colors.reset}\n`);

        console.log(`${colors.bright}Status:${colors.reset} ${result.emoji} ${result.status.toUpperCase()}\n`);

        if (result.error) {
            console.log(`${colors.red}Error: ${result.error}${colors.reset}\n`);
            return;
        }

        const data = result.data;

        // Agent Result
        console.log(`${colors.bright}${colors.blue}Agent Result:${colors.reset}`);
        console.log(`  Status: ${data.agent_result?.status || 'N/A'}`);
        if (data.agent_result?.error) {
            console.log(`  Error: ${colors.red}${data.agent_result.error}${colors.reset}`);
        }
        if (data.agent_result?.diff) {
            const diffLines = data.agent_result.diff.split('\n').length;
            console.log(`  Diff: ${diffLines} lines of changes`);
        }
        console.log();

        // Eval Result
        if (data.eval_result) {
            console.log(`${colors.bright}${colors.blue}Evaluation Result:${colors.reset}`);
            console.log(`  Status: ${data.eval_result.status || 'N/A'}`);
            
            if (data.eval_result.test_results) {
                const tests = data.eval_result.test_results;
                const passedTests = tests.filter(t => t.status === 'pass').length;
                const failedTests = tests.filter(t => t.status === 'fail').length;
                
                console.log(`  Tests: ${colors.green}${passedTests} passed${colors.reset}, ${colors.red}${failedTests} failed${colors.reset} (${tests.length} total)`);
                
                console.log(`\n${colors.bright}Test Details:${colors.reset}`);
                tests.forEach(test => {
                    const emoji = test.status === 'pass' ? '✅' : '❌';
                    const color = test.status === 'pass' ? colors.green : colors.red;
                    console.log(`  ${emoji} ${color}${test.name}${colors.reset} (${test.category})`);
                });
            }
        } else {
            console.log(`${colors.yellow}No evaluation result available${colors.reset}`);
        }

        // Show full logs if available
        if (data.agent_result?.logs) {
            console.log(`\n${colors.bright}${colors.blue}Agent Logs:${colors.reset}`);
            const logLines = data.agent_result.logs.split('\n');
            logLines.forEach(line => {
                if (line.trim()) {
                    // Color code different log levels
                    let coloredLine = line;
                    if (line.includes('[ERROR]') || line.includes('Error') || line.includes('Failed')) {
                        coloredLine = `${colors.red}${line}${colors.reset}`;
                    } else if (line.includes('[WARN]') || line.includes('Warning')) {
                        coloredLine = `${colors.yellow}${line}${colors.reset}`;
                    } else if (line.includes('[AGENT_RUNNER]')) {
                        coloredLine = `${colors.cyan}${line}${colors.reset}`;
                    } else if (line.includes('[AGENT]')) {
                        coloredLine = `${colors.magenta}${line}${colors.reset}`;
                    } else {
                        coloredLine = `${colors.dim}${line}${colors.reset}`;
                    }
                    console.log(`  ${coloredLine}`);
                }
            });
        }
    }

    cleanup() {
        if (process.stdin.isTTY) {
            process.stdin.setRawMode(false);
        }
        console.log(colors.clearScreen + colors.moveCursor);
        console.log(`${colors.green}Thanks for using Results Viewer!${colors.reset}`);
    }

    start() {
        console.log(colors.clearScreen + colors.moveCursor);
        this.render();
    }
}

// Main execution
const runsDirectory = process.argv[2] || './runs2';

if (!fs.existsSync(runsDirectory)) {
    console.error(`${colors.red}Error: Directory '${runsDirectory}' does not exist${colors.reset}`);
    console.log(`Usage: node results_viewer.js [directory]`);
    process.exit(1);
}

const viewer = new ResultsViewer(runsDirectory);
viewer.start();