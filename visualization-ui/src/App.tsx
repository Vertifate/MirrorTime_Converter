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

  // 启动处理流程
  const startPipeline = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/pipeline/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          preset: 'default',
          input: 'sample_video.mp4'
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

  // 获取状态图标
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return '⚙️'
      case 'completed':
        return '✅'
      case 'failed':
        return '❌'
      default:
        return '⏳'
    }
  }

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return '#3b82f6'
      case 'completed':
        return '#10b981'
      case 'failed':
        return '#ef4444'
      default:
        return '#6b7280'
    }
  }

  return (
    <div className="app">
      {/* 顶部标题栏 */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">
            <span className="title-icon">🎬</span>
            MirrorTime Converter
          </h1>
          <div className="header-status">
            <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
            <span className="status-text">
              {isConnected ? '已连接' : '未连接'}
            </span>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="main-content">
        {/* 控制面板 */}
        <div className="control-panel">
          <h2 className="panel-title">流程控制</h2>
          <div className="control-buttons">
            <button
              className="btn btn-primary"
              onClick={startPipeline}
              disabled={isProcessing || !isConnected}
            >
              {isProcessing ? '⚙️ 处理中...' : '▶️ 启动处理'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={stopPipeline}
              disabled={!isProcessing}
            >
              ⏸️ 停止
            </button>
          </div>
        </div>

        {/* 处理阶段列表 */}
        <div className="stages-panel">
          <h2 className="panel-title">处理阶段</h2>
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
                    <span className="stage-icon">{getStatusIcon(stage.status)}</span>
                    <h3 className="stage-name">{stage.name}</h3>
                  </div>
                  <div className="stage-status" style={{ color: getStatusColor(stage.status) }}>
                    {stage.status === 'running' && `${stage.progress}%`}
                    {stage.status === 'completed' && '完成'}
                    {stage.status === 'failed' && '失败'}
                    {stage.status === 'pending' && '等待'}
                  </div>
                </div>

                {/* 进度条 */}
                {stage.status === 'running' && (
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${stage.progress}%` }}
                    />
                  </div>
                )}

                {/* 状态消息 */}
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
