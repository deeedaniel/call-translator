from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

LANG_TO_BCP47: dict[str, str] = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "pt": "pt-BR",
    "zh": "zh-CN",
    "ar": "ar-SA",
    "ru": "ru-RU",
}


class TranslationWorker:
    """Sends text to the NVIDIA NIM Riva Translate 1.6b API via gRPC.

    Mirrors the ``ASRWorker`` pattern: lazy gRPC connection, blocking
    ``translate_sync`` method designed for ``run_in_executor`` usage.
    """

    def __init__(
        self,
        nvidia_api_key: str,
        function_id: str = "0778f2eb-b64d-45e7-acae-7dd9b9b35b4d",
        grpc_endpoint: str = "grpc.nvcf.nvidia.com:443",
    ) -> None:
        self._api_key = nvidia_api_key
        self._function_id = function_id
        self._grpc_endpoint = grpc_endpoint
        self._auth: Optional[Any] = None
        self._nmt_service: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Connection                                                          #
    # ------------------------------------------------------------------ #

    def _ensure_connection(self) -> None:
        """Lazily initialise the gRPC channel and NMT service."""
        if self._nmt_service is not None:
            return

        import riva.client

        metadata = [
            ("function-id", self._function_id),
            ("authorization", f"Bearer {self._api_key}"),
        ]

        self._auth = riva.client.Auth(
            ssl_cert=None,
            use_ssl=True,
            uri=self._grpc_endpoint,
            metadata_args=metadata,
        )
        self._nmt_service = riva.client.NeuralMachineTranslationService(self._auth)
        logger.info(
            "Translation worker connected to NIM endpoint %s",
            self._grpc_endpoint,
        )

    # ------------------------------------------------------------------ #
    # Language normalisation                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_lang(code: str) -> str:
        """Convert a bare ISO 639-1 code to BCP-47 as required by Riva NMT."""
        return LANG_TO_BCP47.get(code, code)

    # ------------------------------------------------------------------ #
    # Translation                                                          #
    # ------------------------------------------------------------------ #

    def translate_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Blocking call: send text to Riva NMT, return translated string.

        Intended to be called via ``asyncio.get_event_loop().run_in_executor()``.
        """
        self._ensure_connection()
        assert self._nmt_service is not None

        response = self._nmt_service.translate(
            [text],
            "",
            self.normalize_lang(source_lang),
            self.normalize_lang(target_lang),
        )
        return response.translations[0].text
