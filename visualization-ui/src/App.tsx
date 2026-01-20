import { useState, useEffect } from 'react'
import './App.css'

// 处理阶段定义
interface Stage {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message: string
  start_time: string | null
  end_time: string | null
}

interface PipelineState {
  current_stage: string | null
  stages: Stage[]
}

function App() {
  const [pipelineState, setPipelineState] = useState<PipelineState>({
    current_stage: null,
    stages: []
  })
  const [isConnected, setIsConnected] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [projectDirectory, setProjectDirectory] = useState<string>('')

  // #WDD [2026-01-19] [Load project directory from localStorage on mount]
  useEffect(() => {
    const savedDirectory = localStorage.getItem('projectDirectory')
    if (savedDirectory) {
      setProjectDirectory(savedDirectory)
    }
  }, [])

  // #WDD [2026-01-19] [Save project directory to localStorage when changed]
  const handleDirectoryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const directory = e.target.value
    setProjectDirectory(directory)
    localStorage.setItem('projectDirectory', directory)
  }

  // WebSocket 连接
  useEffect(() => {
    const connectWebSocket = () => {
      const websocket = new WebSocket('ws://localhost:8000/ws')

      websocket.onopen = () => {
        console.log('✅ WebSocket 已连接')
        setIsConnected(true)
      }

      websocket.onmessage = (event) => {
        const message = JSON.parse(event.data)
        console.log('📨 收到消息:', message)

        if (message.type === 'initial_state' || message.type === 'status_update') {
          setPipelineState(message.data)
        } else if (message.type === 'pipeline_started') {
          setIsProcessing(true)
          setPipelineState(message.data)
        } else if (message.type === 'stage_started' || message.type === 'progress_update') {
          setPipelineState(message.data.pipeline || message.data)
        } else if (message.type === 'pipeline_completed') {
          setIsProcessing(false)
          setPipelineState(message.data)
        }
      }

      websocket.onclose = () => {
        console.log('❌ WebSocket 已断开')
        setIsConnected(false)
        // 3秒后重连
        setTimeout(connectWebSocket, 3000)
      }

      websocket.onerror = (error) => {
        console.error('WebSocket 错误:', error)
      }

      setWs(websocket)
    }

    connectWebSocket()

    return () => {
      ws?.close()
    }
  }, [])

  // Start pipeline with project directory
  // #WDD [2026-01-19] [Include project directory in pipeline config]
  const startPipeline = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/pipeline/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          preset: 'default',
          input: 'sample_video.mp4',
          project_directory: projectDirectory
        }),
      })

      if (response.ok) {
        console.log('🚀 流程已启动')
      }
    } catch (error) {
      console.error('启动失败:', error)
    }
  }

  // 停止处理流程
  const stopPipeline = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/pipeline/stop', {
        method: 'POST',
      })

      if (response.ok) {
        setIsProcessing(false)
        console.log('⏸️ 流程已停止')
      }
    } catch (error) {
      console.error('停止失败:', error)
    }
  }

  // Get status text
  const getStatusText = (status: string, progress: number) => {
    switch (status) {
      case 'running':
        return `${progress}%`
      case 'completed':
        return 'Completed'
      case 'failed':
        return 'Failed'
      default:
        return 'Pending'
    }
  }

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return '#0066ff'
      case 'completed':
        return '#00cc66'
      case 'failed':
        return '#ff3333'
      default:
        return '#666666'
    }
  }

  return (
    <div className="app">
      {/* 顶部标题栏 */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">
            MirrorTime Converter
          </h1>
          <div className="header-status">
            <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
            <span className="status-text">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Full-width Control Bar */}
      <div className="control-bar">
        <div className="control-bar-content">
          {/* Project Directory Input */}
          <div className="directory-selector">
            <span className="input-prefix">PROJECT_DIR</span>
            <input
              type="text"
              className="directory-input"
              value={projectDirectory}
              onChange={handleDirectoryChange}
              placeholder="/path/to/project"
              disabled={isProcessing}
            />
          </div>

          <div className="control-separator"></div>

          <div className="control-buttons">
            <button
              className="btn btn-primary"
              onClick={startPipeline}
              disabled={isProcessing || !isConnected}
            >
              {isProcessing ? 'Running...' : 'Start'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={stopPipeline}
              disabled={!isProcessing}
            >
              Stop
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="main-content">
        {/* Stages Panel */}
        <div className="stages-panel">
          <h2 className="panel-title">Stages</h2>
          <div className="stages-list">
            {pipelineState.stages.map((stage, index) => (
              <div
                key={stage.id}
                className={`stage-card ${stage.status} ${pipelineState.current_stage === stage.id ? 'active' : ''
                  }`}
              >
                <div className="stage-header">
                  <div className="stage-info">
                    <span className="stage-number">{index + 1}</span>
                    <h3 className="stage-name">{stage.name}</h3>
                  </div>
                  <div className="stage-status" style={{ color: getStatusColor(stage.status) }}>
                    {getStatusText(stage.status, stage.progress)}
                  </div>
                </div>

                {/* Progress Bar */}
                {stage.status === 'running' && (
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${stage.progress}%` }}
                    />
                  </div>
                )}

                {/* Status Message */}
                {stage.message && (
                  <div className="stage-message">{stage.message}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
  ```
