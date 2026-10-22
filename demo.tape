Output docs/demo.gif
Set FontSize 18
Set Width 1400
Set Height 900
Set TypingSpeed 200ms
Set Padding 20

# Start server in background
Type "make run &" Enter
Sleep 20s

# Run the interactive chat CLI
Type "make chat" Enter
Sleep 3s

# Show rule normalization + grounded retrieval
Type "explain PRE31-C" Enter
Sleep 20s

# Show unhyphenated form still works
Type "pre31c" Enter
Sleep 20s

# Show strict grounding - refuse out of corpus question
Type "Explain MCP and ACP protocols" Enter
Sleep 10s

# Exit cleanly
Type "exit" Enter
Sleep 5s

# Clean up background process
Type "pkill -f 'uvicorn' || pkill -f 'app.py'" Enter
Sleep 5s
