# Jetpack Compose 开发规范

## 组件设计原则

### 1. 无状态 Composable（Stateless Composable）
所有 UI 组件应设计为无状态，将状态提升到 ViewModel 层。

```kotlin
// ✅ 推荐
@Composable
fun VideoCard(
    video: VideoItem,
    isPlaying: Boolean,
    onPlay: () -> Unit,
    onLike: (String) -> Unit,
) {
    // 纯 UI 渲染
}

// ❌ 不推荐 - 在 Composable 内部持有业务状态
@Composable
fun VideoCard(videoId: String) {
    val viewModel: VideoViewModel = hiltViewModel()
    // 直接依赖 ViewModel，难以测试
}
```

### 2. 状态管理
使用 `StateFlow` + `collectAsStateWithLifecycle` 进行响应式 UI 更新。

```kotlin
@HiltViewModel
class FeedViewModel @Inject constructor(
    private val feedRepository: FeedRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow<FeedUiState>(FeedUiState.Loading)
    val uiState: StateFlow<FeedUiState> = _uiState.asStateFlow()

    fun loadFeed(page: Int) {
        viewModelScope.launch {
            feedRepository.getFeed(page)
                .onSuccess { items ->
                    _uiState.value = FeedUiState.Success(items)
                }
                .onFailure { error ->
                    _uiState.value = FeedUiState.Error(error.message ?: "未知错误")
                }
        }
    }
}

sealed class FeedUiState {
    object Loading : FeedUiState()
    data class Success(val items: List<VideoItem>) : FeedUiState()
    data class Error(val message: String) : FeedUiState()
}
```

### 3. Feed 列表实现
短剧 Feed 使用 `LazyColumn` + Paging3 实现无限滚动。

```kotlin
@Composable
fun FeedScreen(
    uiState: FeedUiState,
    onLoadMore: () -> Unit,
    onVideoPlay: (String) -> Unit,
) {
    val lazyPagingItems = rememberLazyListState()

    LazyColumn(state = lazyPagingItems) {
        when (uiState) {
            is FeedUiState.Loading -> item { LoadingIndicator() }
            is FeedUiState.Success -> {
                items(
                    items = uiState.items,
                    key = { it.videoId }
                ) { video ->
                    VideoCard(
                        video = video,
                        onPlay = { onVideoPlay(video.videoId) }
                    )
                }
            }
            is FeedUiState.Error -> item { ErrorView(uiState.message) }
        }
    }
}
```

## 性能优化

### 避免不必要重组
- 使用 `key` 参数标识列表项
- 将 Lambda 用 `remember` 包裹避免每次重组创建新对象
- 使用 `derivedStateOf` 计算派生状态

```kotlin
// 避免 Lambda 重组
val onLikeStable = remember(videoId) {
    { viewModel.toggleLike(videoId) }
}

// 派生状态
val showScrollToTop by remember {
    derivedStateOf { listState.firstVisibleItemIndex > 5 }
}
```

## 测试规范
- UI 组件使用 Compose Test 进行截图测试
- ViewModel 使用 `TestCoroutineDispatcher` 进行单元测试
