class ChatBot {
    constructor() {
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.loading = document.getElementById('loading');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.inputContainer = document.querySelector('.input-container');
        this.chatHistory = [];
        this.isSystemOnline = false;
        this.statusCheckInterval = null;
        this.currentStatusInterval = null;
        
        this.init();
    }
    
    init() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.startStatusMonitoring();
        this.checkSystemStatus();
        
        setTimeout(() => {
            if (this.isSystemOnline) {
                this.messageInput.focus();
            }
        }, 1000);
    }
    
    async sendMessage(message = null) {
        const userMessage = message || this.messageInput.value.trim();
        if (!userMessage || !this.isSystemOnline) {
            if (!this.isSystemOnline) {
                this.showSystemMessage('System is not online. Please wait for the system to be ready.');
            }
            return;
        }
        
        // Add user message
        this.addMessage(userMessage, 'user');
        this.chatHistory.push({
            sender: 'user',
            content: userMessage
        });
        this.messageInput.value = '';
        this.setLoading(true);
        let queryId = null;
        let statusInterval = null;
        
        try {
            const responsePromise = fetch('http://localhost:8080/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    question: userMessage,
                    chat_history: this.chatHistory 
                })
            });
            
            let statusCheckCount = 0;
            const tempStatusInterval = setInterval(async () => {
                statusCheckCount++;
                if (statusCheckCount > 60) {
                    clearInterval(tempStatusInterval);
                    return;
                }
                
                try {
                    const statusListResponse = await fetch('http://localhost:8080/api/status/latest');
                    if (statusListResponse.ok) {
                        const status = await statusListResponse.json();
                        this.updateLoadingStatus(status.message || 'Processing...');
                    }
                } catch (e) {
                    // Silently continue
                }
            }, 500);
            
            const response = await responsePromise;
            const data = await response.json();
            
            clearInterval(tempStatusInterval);
            
            if (response.ok) {
                this.addMessage(data.answer, 'bot', data.sources);
                this.chatHistory.push({
                    sender: 'bot',
                    content: data.answer
                });
            } else {
                this.addMessage(`Error: ${data.error}`, 'bot');
            }
        } catch (error) {
            this.addMessage(`Network error: ${error.message}`, 'bot');
            if (error.message.includes('fetch') || error.message.includes('network')) {
                this.updateSystemStatus('offline', 'Connection failed');
            }
        } finally {
            if (this.currentStatusInterval) {
                clearInterval(this.currentStatusInterval);
            }
            this.setLoading(false);
            if (this.isSystemOnline) {
                this.messageInput.focus();
            }
        }
    }
    
    addMessage(content, sender, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        
        if (sender === 'user') {
            avatar.innerHTML = `
                <svg viewBox="0 0 40 40" width="40" height="40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="20" cy="20" r="20" fill="#4F46E5"/>
                    <circle cx="20" cy="16" r="6" fill="white"/>
                    <path d="M20 26c-6 0-11 3-11 6v2c0 1 1 2 2 2h18c1 0 2-1 2-2v-2c0-3-5-6-11-6z" fill="white"/>
                </svg>
            `;
        } else {
            avatar.innerHTML = `
                <svg viewBox="0 0 40 40" width="40" height="40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="20" cy="20" r="20" fill="#10B981"/>
                    <rect x="12" y="10" width="16" height="14" rx="2" fill="white"/>
                    <circle cx="16" cy="16" r="2" fill="#10B981"/>
                    <circle cx="24" cy="16" r="2" fill="#10B981"/>
                    <rect x="14" y="20" width="12" height="2" rx="1" fill="#10B981"/>
                    <rect x="8" y="14" width="3" height="6" rx="1.5" fill="white"/>
                    <rect x="29" y="14" width="3" height="6" rx="1.5" fill="white"/>
                    <circle cx="20" cy="30" r="2" fill="white"/>
                </svg>
            `;
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.innerHTML = content.replace(/\n/g, '<br>');
        contentDiv.appendChild(textDiv);
        
        // Add sources if provided
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            
            // Create collapsible header
            const sourcesHeader = document.createElement('div');
            sourcesHeader.className = 'sources-header';
            
            const sourcesTitle = document.createElement('div');
            sourcesTitle.className = 'sources-title';
            sourcesTitle.innerHTML = `
                <span>Sources</span>
                <span class="sources-count">${sources.length}</span>
            `;
            
            const sourcesToggle = document.createElement('button');
            sourcesToggle.className = 'sources-toggle';
            sourcesToggle.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6,9 12,15 18,9"></polyline>
                </svg>
            `;
            
            sourcesHeader.appendChild(sourcesTitle);
            sourcesHeader.appendChild(sourcesToggle);
            sourcesDiv.appendChild(sourcesHeader);
            
            // Create collapsible content
            const sourcesContent = document.createElement('div');
            sourcesContent.className = 'sources-content';
            
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';
            
            sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                
                const sourceContent = document.createElement('div');
                sourceContent.className = 'source-content';
                
                const iconSpan = document.createElement('span');
                iconSpan.className = 'source-icon';
                iconSpan.innerHTML = source.icon;
                
                const textSpan = document.createElement('span');
                textSpan.className = 'source-text';
                textSpan.textContent = `${source.title || 'Unknown'} (${source.type || 'File'})`;
                
                const scoreSpan = document.createElement('span');
                scoreSpan.className = 'source-confidence';
                
                // Handle score display
                let scoreText;
                if (source.score !== undefined) {
                    scoreText = `${source.score}% match`;
                } else if (source.confidence) {
                    scoreText = `${source.confidence}% match`;
                } else if (source.relevance) {
                    scoreText = `${source.relevance} relevance`;
                } else {
                    scoreText = '75% match';
                }
                
                scoreSpan.textContent = scoreText;
                
                // Add view button if URL is available
                const actionContainer = document.createElement('div');
                actionContainer.className = 'source-actions';
                
                if (source.view_url) {
                    const viewButton = document.createElement('a');
                    viewButton.href = source.view_url;
                    viewButton.target = '_blank';
                    viewButton.className = 'view-button';
                    viewButton.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg> View';
                    actionContainer.appendChild(viewButton);
                }
                
                sourceContent.appendChild(iconSpan);
                sourceContent.appendChild(textSpan);
                sourceContent.appendChild(scoreSpan);
                if (source.view_url) {
                    sourceContent.appendChild(actionContainer);
                }
                sourceItem.appendChild(sourceContent);
                
                sourcesContainer.appendChild(sourceItem);
            });
            
            sourcesContent.appendChild(sourcesContainer);
            sourcesDiv.appendChild(sourcesContent);
            
            // Add toggle functionality
            const toggleSources = () => {
                const isExpanded = sourcesContent.classList.contains('expanded');
                if (isExpanded) {
                    sourcesContent.classList.remove('expanded');
                    sourcesToggle.classList.remove('expanded');
                } else {
                    sourcesContent.classList.add('expanded');
                    sourcesToggle.classList.add('expanded');
                }
            };
            
            sourcesHeader.addEventListener('click', toggleSources);
            
            contentDiv.appendChild(sourcesDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        
        // Create a wrapper that contains everything for this message
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message-wrapper ${sender}-message-wrapper`;
        messageWrapper.appendChild(messageDiv);
        
        // Add action buttons below the message
        const actionButtons = document.createElement('div');
        actionButtons.className = 'action-buttons-external';

        // Add copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button-small';
        copyButton.title = sender === 'bot' ? 'Copy response' : 'Copy question';
        copyButton.innerHTML = `
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.5">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        `;
        copyButton.addEventListener('click', () => this.copyToClipboard(content, copyButton));
        actionButtons.appendChild(copyButton);

        // Add retry button for bot messages
        if (sender === 'bot') {
            const retryButton = document.createElement('button');
            retryButton.className = 'retry-button-small';
            retryButton.title = 'Try again';
            retryButton.innerHTML = `
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.5">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                </svg>
            `;
            retryButton.addEventListener('click', () => this.retryLastQuestion());
            actionButtons.appendChild(retryButton);
        }
        
        messageWrapper.appendChild(actionButtons);
        
        this.chatContainer.appendChild(messageWrapper);
        this.scrollToBottom();
    }
    
    retryMessage(message) {
        this.sendMessage(message);
    }
    
    retryLastQuestion() {
        // Find the last user message in chat history
        for (let i = this.chatHistory.length - 1; i >= 0; i--) {
            if (this.chatHistory[i].sender === 'user') {
                this.sendMessage(this.chatHistory[i].content);
                break;
            }
        }
    }
    
    setLoading(isLoading) {
        this.loading.style.display = isLoading ? 'flex' : 'none';
        this.sendButton.disabled = isLoading;
        this.messageInput.disabled = isLoading;
        
        if (isLoading) {
            this.scrollToBottom();
        }
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        }, 100);
    }
    
    async copyToClipboard(text, button) {
        try {
            // Clean text for copying (remove HTML tags)
            const cleanText = text.replace(/<br>/g, '\n').replace(/<[^>]*>/g, '');
            await navigator.clipboard.writeText(cleanText);
            
            // Show feedback
            const originalText = button.innerHTML;
            button.innerHTML = `
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20,6 9,17 4,12"></polyline>
                </svg>
            `;
            button.classList.add('copied');
            
            // Reset after 2 seconds
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('copied');
            }, 2000);
            
        } catch (err) {
            console.error('Failed to copy text: ', err);
            // Fallback for older browsers
            button.textContent = 'Copy failed';
            setTimeout(() => {
                button.innerHTML = `
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                `;
            }, 2000);
        }
    }
    
    startStatusMonitoring() {
        // Check status every 10 seconds
        this.statusCheckInterval = setInterval(() => {
            this.checkSystemStatus();
        }, 10000);
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('http://localhost:8080/api/health', {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'healthy' && data.ready) {
                    this.updateSystemStatus('online', 'Online');
                } else if (data.status === 'loading') {
                    this.updateSystemStatus('loading', 'Starting up...');
                } else {
                    this.updateSystemStatus('connecting', 'Connecting...');
                }
            } else if (response.status === 503) {
                // Service unavailable - system is loading
                try {
                    const data = await response.json();
                    this.updateSystemStatus('loading', data.message || 'Starting up...');
                } catch {
                    this.updateSystemStatus('loading', 'Starting up...');
                }
            } else {
                this.updateSystemStatus('connecting', 'Connecting...');
            }
        } catch (error) {
            this.updateSystemStatus('offline', 'Offline');
        }
    }
    
    updateSystemStatus(status, text) {
        // Remove existing status classes
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = text;
        
        // Update system online state
        const wasOnline = this.isSystemOnline;
        this.isSystemOnline = (status === 'online');
        
        // Update UI based on status
        if (this.isSystemOnline) {
            this.inputContainer.classList.remove('disabled');
            this.messageInput.disabled = false;
            this.sendButton.disabled = false;
            this.messageInput.placeholder = 'Ask me anything...';
            
            // Focus input if system just came online
            if (!wasOnline) {
                this.messageInput.focus();
                this.removeSystemMessages();
            }
        } else {
            this.inputContainer.classList.add('disabled');
            this.messageInput.disabled = true;
            this.sendButton.disabled = true;
            this.messageInput.placeholder = 'System is not ready...';
            
            // Show system message if system just went offline
            if (wasOnline && status === 'offline') {
                this.showSystemMessage('System is offline. Reconnecting...');
            } else if (status === 'loading') {
                this.showSystemMessage('System is starting up. Please wait...');
            } else if (status === 'connecting') {
                this.showSystemMessage('Connecting to system...');
            }
        }
    }
    
    showSystemMessage(message) {
        // Remove existing system messages
        this.removeSystemMessages();
        
        const systemDiv = document.createElement('div');
        systemDiv.className = 'system-message';
        systemDiv.textContent = message;
        
        this.chatContainer.appendChild(systemDiv);
        this.scrollToBottom();
    }
    
    removeSystemMessages() {
        const systemMessages = this.chatContainer.querySelectorAll('.system-message');
        systemMessages.forEach(msg => msg.remove());
    }
    
    async checkQueryStatus(queryId) {
        try {
            const response = await fetch(`http://localhost:8080/api/status/${queryId}`);
            if (response.ok) {
                const status = await response.json();
                this.updateLoadingStatus(status.message);
                
                // Stop polling if query is complete
                if (status.step === 'complete') {
                    clearInterval(this.currentStatusInterval);
                }
            }
        } catch (error) {
            // Silently handle errors to avoid spamming
        }
    }
    
    updateLoadingStatus(message) {
        const statusMessage = document.getElementById('statusMessage');
        if (statusMessage) {
            statusMessage.textContent = message;
        }
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});