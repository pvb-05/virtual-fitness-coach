/**
 * Virtual Fitness Coach - Stream & Realtime Controller
 * Quản lý webcam, nén frame, truyền nhận qua WebSocket, vẽ khung xương canvas và cập nhật giao diện.
 */

class WorkoutStreamController {
  constructor(options = {}) {
    // DOM Elements
    this.video = document.getElementById(options.videoId || 'videoElement');
    this.canvas = document.getElementById(options.canvasId || 'overlayCanvas');
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

    // HUD Elements
    this.repCountEl = document.getElementById('repCountDisplay');
    this.stateBadgeEl = document.getElementById('stateBadge');
    this.progressFillEl = document.getElementById('depthProgressFill');
    this.progressTextEl = document.getElementById('depthProgressText');
    this.feedbackBannerEl = document.getElementById('feedbackBanner');
    this.feedbackTextEl = document.getElementById('feedbackText');
    this.kneeAngleEl = document.getElementById('kneeAngleDisplay');
    this.latencyEl = document.getElementById('latencyDisplay');
    this.fpsEl = document.getElementById('fpsDisplay');
    this.sessionTimerEl = document.getElementById('sessionTimer');
    this.connectionStatusEl = document.getElementById('connectionStatus');

    // Nút điều khiển
    this.startBtn = document.getElementById('startWorkoutBtn');
    this.pauseBtn = document.getElementById('pauseWorkoutBtn');
    this.stopBtn = document.getElementById('stopWorkoutBtn');
    this.muteBtn = document.getElementById('toggleAudioBtn');

    // Quản lý âm thanh TTS
    this.currentAudio = null;
    this.isMuted = false;
    this.lastPlayedAudioUrl = null;

    // Quản lý lịch sử lỗi của buổi tập (để lưu vào SQLite)
    this.sessionErrors = [];
    this.lastErrorTrackTime = 0;

    // Trạng thái stream & kết nối
    this.stream = null;
    this.ws = null;
    this.isRunning = false;
    this.isPaused = false;
    this.isWaitingForResponse = false;
    this.lastFrameTime = 0;
    this.targetFps = 25; // Giới hạn tốc độ gửi frame để không làm nghẽn backend
    this.frameInterval = 1000 / this.targetFps;

    // Bộ đếm FPS & Thời lượng
    this.fpsCounter = 0;
    this.fpsLastCalc = performance.now();
    this.sessionStartTime = null;
    this.timerInterval = null;
    this.elapsedSeconds = 0;

    // Canvas ẩn để capture và nén frame
    this.captureCanvas = document.createElement('canvas');
    this.captureCanvas.width = 640;
    this.captureCanvas.height = 480;
    this.captureCtx = this.captureCanvas.getContext('2d');

    // MediaPipe Skeleton Connections
    this.skeletonConnections = [
      [11, 12], [11, 23], [12, 24], [23, 24],
      [23, 25], [24, 26], [25, 27], [26, 28],
      [27, 29], [28, 30], [29, 31], [30, 32],
      [11, 13], [13, 15], [12, 14], [14, 16]
    ];

    this.bindEvents();
  }

  bindEvents() {
    if (this.startBtn) this.startBtn.addEventListener('click', () => this.startSession());
    if (this.pauseBtn) this.pauseBtn.addEventListener('click', () => this.togglePause());
    if (this.stopBtn) this.stopBtn.addEventListener('click', () => this.stopSession());
    if (this.muteBtn) this.muteBtn.addEventListener('click', () => this.toggleMute());

    window.addEventListener('beforeunload', () => {
      this.cleanup();
    });
  }

  /**
   * Khởi động buổi tập: Bật camera và kết nối WebSocket
   */
  async startSession() {
    if (this.isRunning) return;

    this.updateConnectionStatus('connecting', 'Đang kết nối camera & máy chủ...');

    try {
      // 1. Khởi tạo Webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play();

      // Đồng bộ kích thước canvas overlay theo video
      this.video.onloadedmetadata = () => {
        this.resizeCanvas();
      };
      this.resizeCanvas();

      // 2. Kết nối WebSocket
      this.initWebSocket();

      this.isRunning = true;
      this.isPaused = false;
      this.sessionStartTime = Date.now() - (this.elapsedSeconds * 1000);
      this.startTimer();

      // Cập nhật nút điều khiển
      if (this.startBtn) this.startBtn.classList.add('hidden');
      if (this.pauseBtn) this.pauseBtn.classList.remove('hidden');
      if (this.stopBtn) this.stopBtn.classList.remove('hidden');

    } catch (err) {
      console.error('Lỗi khởi động phòng tập:', err);
      this.updateConnectionStatus('error', 'Không thể mở webcam: ' + err.message);
      alert('Không thể kết nối camera: ' + err.message);
    }
  }

  resizeCanvas() {
    if (!this.canvas || !this.video) return;
    const rect = this.video.getBoundingClientRect();
    if (rect.width && rect.height) {
      this.canvas.width = rect.width;
      this.canvas.height = rect.height;
    }
  }

  /**
   * Khởi tạo kết nối WebSocket với Flask Backend
   */
  initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket đã kết nối thành công tới:', wsUrl);
      this.updateConnectionStatus('connected', 'Trực tiếp • AI Sẵn Sàng');
      this.isWaitingForResponse = false;
      this.scheduleNextFrame();
    };

    this.ws.onmessage = (event) => {
      this.isWaitingForResponse = false; // Nhận được phản hồi, sẵn sàng gửi frame tiếp theo
      try {
        const data = JSON.parse(event.data);
        this.handleBackendResponse(data);
      } catch (err) {
        console.error('Lỗi phân tích cú pháp JSON phản hồi:', err);
      }
    };

    this.ws.onerror = (err) => {
      console.error('Lỗi kết nối WebSocket:', err);
      this.updateConnectionStatus('error', 'Lỗi WebSocket');
    };

    this.ws.onclose = () => {
      console.log('WebSocket đã đóng');
      if (this.isRunning) {
        this.updateConnectionStatus('disconnected', 'Mất kết nối máy chủ');
      }
    };
  }

  /**
   * Cơ chế gửi frame: Chỉ gửi khi backend đã xử lý xong frame trước (Ping-pong gating)
   * và tuân thủ chu kỳ targetFps để triệt tiêu độ trễ tích lũy.
   */
  scheduleNextFrame() {
    if (!this.isRunning || this.isPaused || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const now = performance.now();
    const elapsed = now - this.lastFrameTime;

    if (!this.isWaitingForResponse && elapsed >= this.frameInterval) {
      this.captureAndSendFrame();
      this.lastFrameTime = now;
    }

    requestAnimationFrame(() => this.scheduleNextFrame());
  }

  captureAndSendFrame() {
    if (!this.video || this.video.videoWidth === 0) return;

    // Vẽ frame vào canvas ẩn kích thước chuẩn 640x480
    this.captureCtx.drawImage(this.video, 0, 0, 640, 480);

    // Nén JPEG chất lượng 0.65 tối ưu băng thông
    const base64Image = this.captureCanvas.toDataURL('image/jpeg', 0.65);

    const payload = {
      type: 'frame',
      image: base64Image,
      timestamp: performance.now()
    };

    try {
      this.ws.send(JSON.stringify(payload));
      this.isWaitingForResponse = true;

      // Timeout phòng vệ: Nếu sau 400ms backend không phản hồi, cho phép gửi frame mới
      setTimeout(() => {
        if (this.isWaitingForResponse) {
          this.isWaitingForResponse = false;
        }
      }, 400);

    } catch (sendErr) {
      console.warn('Lỗi gửi frame qua WebSocket:', sendErr);
      this.isWaitingForResponse = false;
    }
  }

  /**
   * Xử lý kết quả trả về từ Backend Pipeline
   */
  handleBackendResponse(data) {
    if (data.type === 'handshake_ok') return;

    // 1. Cập nhật số Rep & Hiệu ứng
    if (this.repCountEl && data.rep_count !== undefined) {
      this.repCountEl.textContent = data.rep_count;
      if (data.rep_just_completed) {
        this.triggerRepCompletionEffect();
      }
    }

    // 2. Trạng thái State Machine
    if (this.stateBadgeEl && data.state) {
      this.updateStateBadge(data.state);
    }

    // 3. Góc gối & Tiến độ độ sâu
    if (this.kneeAngleEl && data.current_knee_angle !== undefined) {
      this.kneeAngleEl.textContent = `${Math.round(data.current_knee_angle)}°`;
    }

    if (this.progressFillEl && data.progress_pct !== undefined) {
      const pct = Math.min(100, Math.max(0, data.progress_pct));
      this.progressFillEl.style.width = `${pct}%`;
      if (this.progressTextEl) {
        this.progressTextEl.textContent = `${Math.round(pct)}%`;
      }
      // Đổi màu thanh tiến độ khi đạt độ sâu chuẩn (>= 95%)
      if (pct >= 95) {
        this.progressFillEl.className = 'h-full bg-emerald-500 transition-all duration-100';
      } else {
        this.progressFillEl.className = 'h-full bg-amber-500 transition-all duration-100';
      }
    }

    // 4. Bảng phản hồi tư thế (Feedback Banner) & Giọng nói TTS
    this.updateFeedbackBanner(data);

    // Ghi nhận lỗi vào danh sách của buổi tập (để lưu vào SQLite khi kết thúc)
    if (data.error_code && data.error_code !== "None" && data.error_code !== "ERR_UNKNOWN" && data.error_code !== "ERR_NO_PERSON") {
      const now = Date.now();
      if (now - this.lastErrorTrackTime >= 2000) {
        this.sessionErrors.push({
          error_code: data.error_code,
          error_description: data.feedback_text || "Lỗi tư thế squat",
          duration_seconds: 1.0,
          feedback_text: data.feedback_text,
          audio_path: data.audio_url || ""
        });
        this.lastErrorTrackTime = now;
      }
    }

    // Phát âm thanh hướng dẫn giọng nói nếu có và không bị tắt tiếng
    if (data.audio_url && !this.isMuted && data.audio_url !== this.lastPlayedAudioUrl) {
      this.playVoiceGuidance(data.audio_url);
      this.lastPlayedAudioUrl = data.audio_url;
    }

    // 5. Độ trễ & FPS
    if (this.latencyEl && data.processing_time_ms !== undefined) {
      this.latencyEl.textContent = `${data.processing_time_ms} ms`;
    }
    this.calculateFps();

    // 6. Vẽ khung xương lên Canvas Overlay
    this.drawSkeleton(data);
  }

  updateStateBadge(state) {
    const stateMap = {
      'STANDING': { text: 'Chuẩn bị (Đứng)', class: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40' },
      'DESCENDING': { text: 'Đang hạ người', class: 'bg-amber-500/20 text-amber-400 border-amber-500/40' },
      'BOTTOM': { text: 'Đạt đáy Squat', class: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40' },
      'ASCENDING': { text: 'Đang đứng lên', class: 'bg-blue-500/20 text-blue-400 border-blue-500/40' }
    };
    const s = stateMap[state] || { text: state, class: 'bg-slate-700 text-slate-300 border-slate-600' };
    this.stateBadgeEl.textContent = s.text;
    this.stateBadgeEl.className = `px-3 py-1 rounded-full text-xs font-bold border transition ${s.class}`;
  }

  updateFeedbackBanner(data) {
    if (!this.feedbackBannerEl || !this.feedbackTextEl) return;

    if (!data.pose_detected) {
      this.feedbackTextEl.textContent = 'Chưa phát hiện người. Vui lòng đứng trước camera.';
      this.feedbackBannerEl.className = 'p-3 rounded-xl border flex items-center gap-2 text-xs bg-slate-800 border-slate-700 text-slate-400';
      return;
    }

    if (!data.is_valid) {
      this.feedbackTextEl.textContent = data.error_message || 'Khuất chân hoặc ngoài khung hình. Vui lòng lùi xa camera.';
      this.feedbackBannerEl.className = 'p-3 rounded-xl border flex items-center gap-2 text-xs bg-rose-500/10 border-rose-500/30 text-rose-300';
      return;
    }

    // Tư thế hợp lệ, kiểm tra nhãn KNN
    const label = data.pose_label;
    if (label === 'correct') {
      this.feedbackTextEl.textContent = 'Tư thế squat chuẩn xác! Hãy giữ vững phong độ.';
      this.feedbackBannerEl.className = 'p-3 rounded-xl border flex items-center gap-2 text-xs bg-emerald-500/10 border-emerald-500/30 text-emerald-300';
    } else if (label === 'insufficient_depth') {
      this.feedbackTextEl.textContent = 'Cảnh báo: Hạ hông sâu hơn nữa để đùi song song với mặt đất.';
      this.feedbackBannerEl.className = 'p-3 rounded-xl border flex items-center gap-2 text-xs bg-amber-500/10 border-amber-500/30 text-amber-300';
    } else if (label === 'knee_valgus') {
      this.feedbackTextEl.textContent = 'Cảnh báo: Đầu gối bị chụm vào trong! Mở rộng gối hướng theo mũi chân.';
      this.feedbackBannerEl.className = 'p-3 rounded-xl border flex items-center gap-2 text-xs bg-rose-500/10 border-rose-500/30 text-rose-300';
    } else {
      this.feedbackTextEl.textContent = 'Tư thế sẵn sàng. Bắt đầu bài tập squat.';
      this.feedbackBannerEl.className = 'p-3 rounded-xl border flex items-center gap-2 text-xs bg-slate-800 border-slate-700 text-slate-300';
    }
  }

  /**
   * Vẽ khung xương trực quan trên HTML5 Canvas
   */
  drawSkeleton(data) {
    if (!this.ctx || !this.canvas) return;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (!data.pose_detected || !data.landmarks || data.landmarks.length === 0) {
      return;
    }

    const w = this.canvas.width;
    const h = this.canvas.height;
    const landmarks = data.landmarks;
    const isValid = data.is_valid;

    // Màu sắc: Xanh Emerald khi hợp lệ, Đỏ khi khuất hoặc lỗi
    const boneColor = isValid ? '#10b981' : '#ef4444';
    const jointFill = '#ffffff';

    this.ctx.lineWidth = 3;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    // 1. Vẽ các đoạn xương nối
    for (const [startIdx, endIdx] of this.skeletonConnections) {
      if (startIdx < landmarks.length && endIdx < landmarks.length) {
        const p1 = landmarks[startIdx];
        const p2 = landmarks[endIdx];

        if (p1.visibility >= 0.5 && p2.visibility >= 0.5) {
          // Lưu ý lật ngang X (Mirror) vì video webcam đã được CSS scaleX(-1)
          const x1 = (1.0 - p1.x) * w;
          const y1 = p1.y * h;
          const x2 = (1.0 - p2.x) * w;
          const y2 = p2.y * h;

          this.ctx.beginPath();
          this.ctx.moveTo(x1, y1);
          this.ctx.lineTo(x2, y2);
          this.ctx.strokeStyle = boneColor;
          this.ctx.stroke();
        }
      }
    }

    // 2. Vẽ các điểm khớp chính
    const mainJoints = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28];
    for (const idx of mainJoints) {
      if (idx < landmarks.length) {
        const p = landmarks[idx];
        if (p.visibility >= 0.5) {
          const x = (1.0 - p.x) * w;
          const y = p.y * h;

          // Vòng hào quang phát sáng
          this.ctx.beginPath();
          this.ctx.arc(x, y, 6, 0, 2 * Math.PI);
          this.ctx.fillStyle = boneColor;
          this.ctx.fill();

          // Chấm trắng trung tâm
          this.ctx.beginPath();
          this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
          this.ctx.fillStyle = jointFill;
          this.ctx.fill();
        }
      }
    }
  }

  triggerRepCompletionEffect() {
    if (this.repCountEl) {
      this.repCountEl.classList.add('text-emerald-400', 'scale-125');
      setTimeout(() => {
        this.repCountEl.classList.remove('text-emerald-400', 'scale-125');
      }, 300);
    }
  }

  calculateFps() {
    this.fpsCounter++;
    const now = performance.now();
    if (now - this.fpsLastCalc >= 1000) {
      if (this.fpsEl) {
        this.fpsEl.textContent = `FPS: ${this.fpsCounter}`;
      }
      this.fpsCounter = 0;
      this.fpsLastCalc = now;
    }
  }

  startTimer() {
    if (this.timerInterval) clearInterval(this.timerInterval);
    this.timerInterval = setInterval(() => {
      if (!this.isPaused) {
        this.elapsedSeconds++;
        const hrs = String(Math.floor(this.elapsedSeconds / 3600)).padStart(2, '0');
        const mins = String(Math.floor((this.elapsedSeconds % 3600) / 60)).padStart(2, '0');
        const secs = String(this.elapsedSeconds % 60).padStart(2, '0');
        if (this.sessionTimerEl) {
          this.sessionTimerEl.textContent = `${hrs}:${mins}:${secs}`;
        }
      }
    }, 1000);
  }

  togglePause() {
    this.isPaused = !this.isPaused;
    if (this.pauseBtn) {
      this.pauseBtn.innerHTML = this.isPaused
        ? `<svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /></svg>Tiếp tục`
        : `<svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>Tạm dừng`;
    }
    if (!this.isPaused) {
      this.scheduleNextFrame();
    }
  }

  async stopSession() {
    const totalReps = parseInt(this.repCountEl ? this.repCountEl.textContent : '0', 10) || 0;
    const durationSec = this.elapsedSeconds || 0;
    const durationFormatted = this.sessionTimerEl ? this.sessionTimerEl.textContent : '00:00:00';
    const errorsToSave = [...this.sessionErrors];

    this.cleanup();

    try {
      const res = await fetch('/api/workout/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exercise_type: 'squat',
          duration_seconds: durationSec,
          total_reps: totalReps,
          correct_reps: Math.max(0, totalReps - Math.floor(errorsToSave.length / 2)),
          errors: errorsToSave
        })
      });

      if (res.ok) {
        alert(`Buổi tập đã hoàn thành và lưu vào SQLite!\n- Số Rep: ${totalReps}\n- Thời gian: ${durationFormatted}\n- Số lỗi phát hiện: ${errorsToSave.length}`);
        window.location.href = '/history';
        return;
      }
    } catch (saveErr) {
      console.warn('Lỗi lưu buổi tập:', saveErr);
    }

    alert(`Buổi tập hoàn tất!\n- Số Rep: ${totalReps}\n- Thời gian: ${durationFormatted}`);
    window.location.href = '/history';
  }

  playVoiceGuidance(audioUrl) {
    if (!audioUrl || this.isMuted) return;

    // Dừng âm thanh trước nếu đang phát để tránh nói đè lên nhau
    if (this.currentAudio) {
      try {
        this.currentAudio.pause();
        this.currentAudio.currentTime = 0;
      } catch (e) {
        // bỏ qua
      }
      this.currentAudio = null;
    }

    try {
      this.currentAudio = new Audio(audioUrl);
      this.currentAudio.play().catch(playErr => {
        // Trình duyệt có thể chặn autoplay nếu người dùng chưa tương tác bấm phím/click
        console.log('Audio autoplay prevented:', playErr.message);
      });
    } catch (err) {
      console.warn('Lỗi phát âm thanh TTS:', err);
    }
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.muteBtn) {
      this.muteBtn.innerHTML = this.isMuted
        ? `<svg class="w-4 h-4 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" /></svg>`
        : `<svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /></svg>`;
      this.muteBtn.title = this.isMuted ? "Bật âm thanh hướng dẫn" : "Tắt âm thanh hướng dẫn";
    }
  }

  cleanup() {
    this.isRunning = false;
    this.isPaused = false;
    if (this.timerInterval) clearInterval(this.timerInterval);

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.ctx && this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  updateConnectionStatus(status, text) {
    if (!this.connectionStatusEl) return;
    const colors = {
      connecting: 'bg-amber-400',
      connected: 'bg-emerald-400 animate-pulse',
      disconnected: 'bg-slate-500',
      error: 'bg-rose-500'
    };
    this.connectionStatusEl.innerHTML = `
      <span class="w-2 h-2 rounded-full ${colors[status] || 'bg-slate-400'} mr-1.5"></span>
      <span>${text}</span>
    `;
  }
}

// Khởi tạo controller khi DOM tải xong
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('videoElement')) {
    window.workoutController = new WorkoutStreamController();
  }
});
