"""Comprehensive media scraping and downloading system with robust error handling.

Features:
- Video downloading from all platforms
- Image downloading and processing
- Quality selection (4K, 1080p, 720p, etc.)
- Format conversion
- CDN integration
- Thumbnail generation
- Metadata extraction
- Comprehensive validation
- Resource management
- Error recovery
"""

import asyncio
import hashlib
import logging
import mimetypes
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import aiofiles
import aiohttp
from PIL import Image
from pydantic import BaseModel

from app.core.errors import MediaError, ValidationError
from app.core.models import SourcePlatform
from app.core.retry import retry_with_backoff
from app.core.validation import URLValidator

logger = logging.getLogger(__name__)


class MediaType(str, Enum):
    """Media types."""

    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"


class VideoQuality(str, Enum):
    """Video quality options."""

    ULTRA_HD_4K = "2160p"  # 3840x2160
    FULL_HD = "1080p"      # 1920x1080
    HD = "720p"            # 1280x720
    SD = "480p"            # 854x480
    LOW = "360p"           # 640x360
    MOBILE = "240p"        # 426x240
    AUTO = "auto"          # Best available


class ImageFormat(str, Enum):
    """Image formats."""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    ORIGINAL = "original"


class MediaMetadata(BaseModel):
    """Media metadata."""

    id: UUID
    platform: SourcePlatform
    media_type: MediaType
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[int] = None  # seconds
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None  # bytes
    format: Optional[str] = None
    thumbnail_url: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    download_url: Optional[str] = None
    local_path: Optional[str] = None


class MediaDownloader:
    """Comprehensive media downloader for all platforms.

    Supports:
    - YouTube videos (via yt-dlp)
    - TikTok videos
    - Instagram images/videos
    - Facebook images/videos
    - Reddit images/videos/GIFs
    - Twitter/X images/videos
    - Generic URL downloads
    """

    def __init__(
        self,
        storage_path: str = "./media_storage",
        cdn_enabled: bool = False,
        cdn_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize media downloader.

        Args:
            storage_path: Local storage path for downloaded media
            cdn_enabled: Whether to upload to CDN
            cdn_config: CDN configuration (provider, bucket, credentials)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.cdn_enabled = cdn_enabled
        self.cdn_config = cdn_config or {}

        # Create subdirectories
        for media_type in MediaType:
            (self.storage_path / media_type.value).mkdir(exist_ok=True)

    @retry_with_backoff(max_retries=3, base_delay=2.0, retry_on=(MediaError,))
    async def download_video(
        self,
        url: str,
        platform: SourcePlatform,
        quality: VideoQuality = VideoQuality.FULL_HD,
        extract_audio: bool = False
    ) -> MediaMetadata:
        """Download video from URL with validation and error handling.

        Args:
            url: Video URL
            platform: Source platform
            quality: Desired video quality
            extract_audio: Whether to also extract audio

        Returns:
            Media metadata with local path

        Raises:
            MediaError: If download fails
            ValidationError: If URL is invalid
        """
        # Validate URL
        try:
            URLValidator(url=url)
        except Exception as e:
            raise ValidationError(f"Invalid video URL: {e}")

        logger.info(f"Downloading video from {platform.value}: {url}")

        # Use yt-dlp for video downloading (supports most platforms)
        try:
            import yt_dlp
        except ImportError:
            raise MediaError("yt-dlp not installed. Install with: pip install yt-dlp")

        # Generate unique ID
        media_id = uuid4()
        output_dir = self.storage_path / MediaType.VIDEO.value / str(media_id)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise MediaError(f"Failed to create output directory: {e}")

        # Configure yt-dlp options
        ydl_opts = {
            "format": self._get_format_string(quality),
            "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "writethumbnail": True,
            "writesubtitles": False,
            "socket_timeout": 30,  # 30 second timeout
            "retries": 3,  # Retry 3 times
        }

        # Download video
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # Get downloaded file path
                downloaded_file = ydl.prepare_filename(info)

                # Verify file exists
                if not os.path.exists(downloaded_file):
                    raise MediaError(f"Downloaded file not found: {downloaded_file}")

                # Extract metadata
                metadata = MediaMetadata(
                    id=media_id,
                    platform=platform,
                    media_type=MediaType.VIDEO,
                    url=url,
                    title=info.get("title"),
                    description=info.get("description"),
                    duration=info.get("duration"),
                    width=info.get("width"),
                    height=info.get("height"),
                    file_size=info.get("filesize") or os.path.getsize(downloaded_file),
                    format=info.get("ext"),
                    thumbnail_url=info.get("thumbnail"),
                    author=info.get("uploader"),
                    published_at=self._parse_upload_date(info.get("upload_date")),
                    local_path=downloaded_file
                )

                logger.info(f"Video downloaded successfully: {metadata.title} ({metadata.file_size} bytes)")

        except Exception as e:
            logger.error(f"Video download failed: {e}")
            raise MediaError(f"Failed to download video: {e}")

        # Extract audio if requested
        if extract_audio:
            try:
                await self._extract_audio(downloaded_file, output_dir)
            except Exception as e:
                logger.warning(f"Audio extraction failed: {e}")
                # Don't fail the whole download if audio extraction fails

        # Upload to CDN if enabled
        if self.cdn_enabled:
            try:
                cdn_url = await self._upload_to_cdn(downloaded_file, media_id)
                metadata.download_url = cdn_url
                logger.info(f"Video uploaded to CDN: {cdn_url}")
            except Exception as e:
                logger.warning(f"CDN upload failed: {e}")
                # Don't fail the whole download if CDN upload fails

        return metadata

    async def download_image(
        self,
        url: str,
        platform: SourcePlatform,
        convert_format: ImageFormat = ImageFormat.ORIGINAL,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None
    ) -> MediaMetadata:
        """Download image from URL.

        Args:
            url: Image URL
            platform: Source platform
            convert_format: Convert to this format
            max_width: Maximum width (resize if larger)
            max_height: Maximum height (resize if larger)

        Returns:
            Media metadata with local path
        """
        media_id = uuid4()
        output_dir = self.storage_path / MediaType.IMAGE.value / str(media_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download image
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise MediaError(f"Failed to download image: HTTP {response.status}")

                # Get content type and extension
                content_type = response.headers.get("Content-Type", "")
                ext = mimetypes.guess_extension(content_type) or ".jpg"

                # Save original
                original_path = output_dir / f"original{ext}"
                async with aiofiles.open(original_path, "wb") as f:
                    await f.write(await response.read())

        # Process image
        processed_path = await self._process_image(
            original_path,
            output_dir,
            convert_format,
            max_width,
            max_height
        )

        # Get image dimensions
        with Image.open(processed_path) as img:
            width, height = img.size
            file_size = os.path.getsize(processed_path)

        metadata = MediaMetadata(
            id=media_id,
            platform=platform,
            media_type=MediaType.IMAGE,
            url=url,
            width=width,
            height=height,
            file_size=file_size,
            format=convert_format.value if convert_format != ImageFormat.ORIGINAL else ext.lstrip("."),
            local_path=str(processed_path)
        )

        # Upload to CDN if enabled
        if self.cdn_enabled:
            cdn_url = await self._upload_to_cdn(str(processed_path), media_id)
            metadata.download_url = cdn_url

        return metadata

    async def download_media_batch(
        self,
        urls: List[str],
        platform: SourcePlatform,
        media_type: MediaType = MediaType.IMAGE,
        max_concurrent: int = 5
    ) -> List[MediaMetadata]:
        """Download multiple media files concurrently.

        Args:
            urls: List of media URLs
            platform: Source platform
            media_type: Type of media
            max_concurrent: Maximum concurrent downloads

        Returns:
            List of media metadata
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(url: str) -> Optional[MediaMetadata]:
            async with semaphore:
                try:
                    if media_type == MediaType.VIDEO:
                        return await self.download_video(url, platform)
                    elif media_type == MediaType.IMAGE:
                        return await self.download_image(url, platform)
                    else:
                        raise MediaError(f"Unsupported media type: {media_type}")
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    return None

        results = await asyncio.gather(*[download_with_semaphore(url) for url in urls])
        return [r for r in results if r is not None]

    def _get_format_string(self, quality: VideoQuality) -> str:
        """Get yt-dlp format string for quality.

        Args:
            quality: Desired quality

        Returns:
            Format string
        """
        if quality == VideoQuality.AUTO:
            return "bestvideo+bestaudio/best"
        elif quality == VideoQuality.ULTRA_HD_4K:
            return "bestvideo[height<=2160]+bestaudio/best[height<=2160]"
        elif quality == VideoQuality.FULL_HD:
            return "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
        elif quality == VideoQuality.HD:
            return "bestvideo[height<=720]+bestaudio/best[height<=720]"
        elif quality == VideoQuality.SD:
            return "bestvideo[height<=480]+bestaudio/best[height<=480]"
        elif quality == VideoQuality.LOW:
            return "bestvideo[height<=360]+bestaudio/best[height<=360]"
        elif quality == VideoQuality.MOBILE:
            return "bestvideo[height<=240]+bestaudio/best[height<=240]"
        else:
            return "bestvideo+bestaudio/best"

    async def _process_image(
        self,
        input_path: Path,
        output_dir: Path,
        convert_format: ImageFormat,
        max_width: Optional[int],
        max_height: Optional[int]
    ) -> Path:
        """Process image (resize, convert format).

        Args:
            input_path: Input image path
            output_dir: Output directory
            convert_format: Target format
            max_width: Maximum width
            max_height: Maximum height

        Returns:
            Processed image path
        """
        with Image.open(input_path) as img:
            # Resize if needed
            if max_width or max_height:
                img.thumbnail((max_width or 10000, max_height or 10000), Image.Resampling.LANCZOS)

            # Determine output format
            if convert_format == ImageFormat.ORIGINAL:
                output_format = img.format or "JPEG"
                ext = f".{output_format.lower()}"
            else:
                output_format = convert_format.value.upper()
                ext = f".{convert_format.value}"

            # Save processed image
            output_path = output_dir / f"processed{ext}"

            # Convert RGBA to RGB for JPEG
            if output_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                rgb_img.save(output_path, format=output_format, quality=95, optimize=True)
            else:
                img.save(output_path, format=output_format, quality=95, optimize=True)

        return output_path

    async def _extract_audio(self, video_path: str, output_dir: Path) -> Path:
        """Extract audio from video.

        Args:
            video_path: Video file path
            output_dir: Output directory

        Returns:
            Audio file path
        """
        try:
            import yt_dlp
        except ImportError:
            raise MediaError("yt-dlp not installed")

        audio_path = output_dir / "audio.mp3"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(audio_path.with_suffix("")),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_path])

        return audio_path

    def _parse_upload_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse upload date from yt-dlp format.

        Args:
            date_str: Date string (YYYYMMDD)

        Returns:
            Datetime object
        """
        if not date_str:
            return None

        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            return None

    async def _upload_to_cdn(self, file_path: str, media_id: UUID) -> str:
        """Upload file to CDN.

        Args:
            file_path: Local file path
            media_id: Media ID

        Returns:
            CDN URL
        """
        # This is a placeholder - implement based on your CDN provider
        # Examples: AWS S3, Google Cloud Storage, Cloudflare R2, etc.

        provider = self.cdn_config.get("provider", "s3")

        if provider == "s3":
            return await self._upload_to_s3(file_path, media_id)
        elif provider == "gcs":
            return await self._upload_to_gcs(file_path, media_id)
        elif provider == "cloudflare":
            return await self._upload_to_cloudflare(file_path, media_id)
        else:
            raise MediaError(f"Unsupported CDN provider: {provider}")

    async def _upload_to_s3(self, file_path: str, media_id: UUID) -> str:
        """Upload to AWS S3.

        Args:
            file_path: Local file path
            media_id: Media ID

        Returns:
            S3 URL
        """
        try:
            import boto3
        except ImportError:
            raise MediaError("boto3 not installed. Install with: pip install boto3")

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.cdn_config.get("aws_access_key_id"),
            aws_secret_access_key=self.cdn_config.get("aws_secret_access_key"),
            region_name=self.cdn_config.get("region", "us-east-1")
        )

        bucket = self.cdn_config.get("bucket")
        key = f"media/{media_id}/{Path(file_path).name}"

        s3_client.upload_file(file_path, bucket, key)

        # Return CDN URL
        if "cloudfront_domain" in self.cdn_config:
            return f"https://{self.cdn_config['cloudfront_domain']}/{key}"
        else:
            return f"https://{bucket}.s3.amazonaws.com/{key}"

    async def _upload_to_gcs(self, file_path: str, media_id: UUID) -> str:
        """Upload to Google Cloud Storage.

        Args:
            file_path: Local file path
            media_id: Media ID

        Returns:
            GCS URL
        """
        try:
            from google.cloud import storage
        except ImportError:
            raise MediaError("google-cloud-storage not installed")

        client = storage.Client()
        bucket = client.bucket(self.cdn_config.get("bucket"))
        blob = bucket.blob(f"media/{media_id}/{Path(file_path).name}")

        blob.upload_from_filename(file_path)

        return blob.public_url

    async def _upload_to_cloudflare(self, file_path: str, media_id: UUID) -> str:
        """Upload to Cloudflare R2.

        Args:
            file_path: Local file path
            media_id: Media ID

        Returns:
            R2 URL
        """
        # Cloudflare R2 uses S3-compatible API
        try:
            import boto3
        except ImportError:
            raise MediaError("boto3 not installed")

        s3_client = boto3.client(
            "s3",
            endpoint_url=self.cdn_config.get("endpoint_url"),
            aws_access_key_id=self.cdn_config.get("access_key_id"),
            aws_secret_access_key=self.cdn_config.get("secret_access_key"),
        )

        bucket = self.cdn_config.get("bucket")
        key = f"media/{media_id}/{Path(file_path).name}"

        s3_client.upload_file(file_path, bucket, key)

        # Return public URL
        public_domain = self.cdn_config.get("public_domain")
        return f"https://{public_domain}/{key}"

    async def generate_thumbnail(
        self,
        video_path: str,
        timestamp: int = 5,
        width: int = 320,
        height: int = 180
    ) -> str:
        """Generate thumbnail from video.

        Args:
            video_path: Video file path
            timestamp: Timestamp in seconds
            width: Thumbnail width
            height: Thumbnail height

        Returns:
            Thumbnail path
        """
        try:
            import ffmpeg
        except ImportError:
            raise MediaError("ffmpeg-python not installed. Install with: pip install ffmpeg-python")

        thumbnail_path = Path(video_path).parent / "thumbnail.jpg"

        try:
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .filter("scale", width, height)
                .output(str(thumbnail_path), vframes=1)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise MediaError(f"Failed to generate thumbnail: {e}")

        return str(thumbnail_path)

    def get_media_info(self, file_path: str) -> Dict[str, Any]:
        """Get media file information.

        Args:
            file_path: Media file path

        Returns:
            Media information (duration, resolution, codec, etc.)
        """
        try:
            import ffmpeg
        except ImportError:
            raise MediaError("ffmpeg-python not installed")

        try:
            probe = ffmpeg.probe(file_path)

            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"),
                None
            )
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"),
                None
            )

            info = {
                "duration": float(probe["format"].get("duration", 0)),
                "size": int(probe["format"].get("size", 0)),
                "format": probe["format"].get("format_name"),
            }

            if video_stream:
                info.update({
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "video_codec": video_stream.get("codec_name"),
                    "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                })

            if audio_stream:
                info.update({
                    "audio_codec": audio_stream.get("codec_name"),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                })

            return info
        except ffmpeg.Error as e:
            raise MediaError(f"Failed to get media info: {e}")

