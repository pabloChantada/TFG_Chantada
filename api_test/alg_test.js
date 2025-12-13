/**
 *  REF: https://nodejs.org/api/child_process.html
 */

// Use child_process to run a Python script from Node.js
const { spawn } = require('child_process');

function executePython() {

    const py = spawn('./.venv/bin/python', ['api_test/alg_test.py']);               // Linux/macOS
    // const py = spawn('.\\.venv\\Scripts\\python.exe', ['api_test/alg_test.py']); // Windows

    // Ouput python
    py.stdout.on('data', (data) => {
        console.log('Python output:', data.toString());
    });

    // Error handling
    py.stderr.on('data', (data) => {
        console.error('Python error:', data.toString());
    });

    // When the process ends
    py.on('close', (code) => {
        console.log('Python process ended with code', code);
    });
}
executePython();
