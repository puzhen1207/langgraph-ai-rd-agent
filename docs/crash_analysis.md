# Crash 分析与调试指南

## 常见 Crash 类型

### 1. ExoPlayer OOM（内存溢出）

**症状**
```
java.lang.OutOfMemoryError: Failed to allocate a X byte allocation
    at com.google.android.exoplayer2.video.VideoDecoderGLSurfaceView
```

**根本原因**
- 多个 ExoPlayer 实例未及时释放
- 视频缓存占用过大
- SurfaceView 持有 Activity 引用

**修复方案**
```kotlin
class VideoPlayerManager {
    private var player: ExoPlayer? = null

    fun initPlayer(context: Context): ExoPlayer {
        releasePlayer() // 先释放旧实例
        return ExoPlayer.Builder(context)
            .setVideoScalingMode(C.VIDEO_SCALING_MODE_SCALE_TO_FIT)
            .build()
            .also { player = it }
    }

    fun releasePlayer() {
        player?.run {
            stop()
            clearMediaItems()
            release()
        }
        player = null
    }
}
```

**预防措施**
- 在 `onPause()` 调用 `player.pause()`
- 在 `onDestroy()` 调用 `releasePlayer()`
- 使用 `SimpleExoPlayer` 的 `setHandleAudioBecomingNoisy(true)`

---

### 2. ANR（主线程卡死）

**症状**
```
ANR in com.example.app (com.example.app/.MainActivity)
PID: 12345
Reason: Input dispatching timed out
```

**排查步骤**
1. 抓取 ANR Trace：`adb pull /data/anr/traces.txt`
2. 查找 main 线程状态，定位 `waiting to lock`
3. 找到持有锁的线程

**常见场景**
- SharedPreferences 在主线程读写
- Room 数据库在主线程查询
- 主线程等待网络请求

**修复模板**
```kotlin
// ❌ 错误：主线程 IO
class BadRepository {
    fun getData() = database.query() // 阻塞主线程
}

// ✅ 正确：协程异步
class GoodRepository(
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
) {
    suspend fun getData(): Result<List<Data>> = withContext(ioDispatcher) {
        runCatching { database.query() }
    }
}
```

---

### 3. NullPointerException

**常见于**
- Fragment 生命周期中使用 `view` 或 `binding`
- ViewModel 在 `DESTROYED` 状态后更新 UI

**修复**
```kotlin
// ❌ 错误
class MyFragment : Fragment() {
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        viewModel.data.observe(viewLifecycleOwner) {
            binding.textView.text = it // 若 view 已销毁会 NPE
        }
    }
}

// ✅ 正确：使用 repeatOnLifecycle
lifecycleScope.launch {
    repeatOnLifecycle(Lifecycle.State.STARTED) {
        viewModel.uiState.collect { state ->
            binding.textView.text = state.text
        }
    }
}
```

## 线上监控

### Firebase Crashlytics 集成
```kotlin
// 自定义日志
FirebaseCrashlytics.getInstance().apply {
    setUserId(userId)
    setCustomKey("video_id", currentVideoId)
    log("Player state: ${player?.playbackState}")
}
```

### 性能埋点
```kotlin
val trace = Firebase.performance.newTrace("video_load_time")
trace.start()
// 播放操作
trace.stop()
trace.putMetric("bytes_loaded", bytesLoaded)
```

## 接口与埋点规范

### 埋点字段规范
| 字段名 | 类型 | 说明 |
|--------|------|------|
| event_name | String | 事件名，下划线命名 |
| video_id | String | 视频唯一 ID |
| play_duration | Long | 播放时长（ms） |
| page_name | String | 页面名称 |
| timestamp | Long | 时间戳（ms） |

### 播放事件
- `video_play_start`：开始播放
- `video_play_pause`：暂停
- `video_play_complete`：播完
- `video_play_error`：播放失败，附带 error_code
