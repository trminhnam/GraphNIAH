"""PDF parsing helpers built on Docling."""

from __future__ import annotations

import os
import sysconfig
import time
from pathlib import Path
from typing import Literal

import requests


def split_library_path(value: str | None) -> list[str]:
    if not value:
        return []
    return [entry for entry in value.split(os.pathsep) if entry]


def build_torch_cuda_library_prefix(site_packages_dir: Path) -> list[str]:
    candidates = [
        site_packages_dir / "torch" / "lib",
        site_packages_dir / "nvidia" / "cudnn" / "lib",
        site_packages_dir / "nvidia" / "cuda_runtime" / "lib",
        site_packages_dir / "nvidia" / "cublas" / "lib",
        site_packages_dir / "nvidia" / "cusparse" / "lib",
        site_packages_dir / "nvidia" / "cusparselt" / "lib",
        site_packages_dir / "nvidia" / "cusolver" / "lib",
        site_packages_dir / "nvidia" / "curand" / "lib",
        site_packages_dir / "nvidia" / "cufft" / "lib",
        site_packages_dir / "nvidia" / "nvjitlink" / "lib",
        site_packages_dir / "nvidia" / "cuda_nvrtc" / "lib",
        site_packages_dir / "nvidia" / "cuda_cupti" / "lib",
        site_packages_dir / "nvidia" / "nccl" / "lib",
        site_packages_dir / "nvidia" / "nvtx" / "lib",
        site_packages_dir / "nvidia" / "cufile" / "lib",
        site_packages_dir / "nvidia" / "nvshmem" / "lib",
        site_packages_dir / "nvidia" / "cu13" / "lib",
    ]
    preferred_dirs: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in seen:
            preferred_dirs.append(candidate_str)
            seen.add(candidate_str)
    return preferred_dirs


def prioritize_torch_cuda_library_path(
    ld_library_path: str | None,
    site_packages_dir: Path,
) -> str | None:
    preferred_dirs = build_torch_cuda_library_prefix(site_packages_dir)
    if not preferred_dirs:
        return ld_library_path

    ordered_entries: list[str] = []
    seen: set[str] = set()
    for entry in preferred_dirs + split_library_path(ld_library_path):
        if entry not in seen:
            ordered_entries.append(entry)
            seen.add(entry)
    return os.pathsep.join(ordered_entries)


def configure_torch_cuda_library_path() -> None:
    site_packages_dir = Path(sysconfig.get_paths()["purelib"])
    current_path = os.environ.get("LD_LIBRARY_PATH")
    updated_path = prioritize_torch_cuda_library_path(current_path, site_packages_dir)
    if updated_path is not None and updated_path != current_path:
        os.environ["LD_LIBRARY_PATH"] = updated_path


configure_torch_cuda_library_path()

import torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from loguru import logger


class PDFParser:
    def __init__(
        self,
        pipeline_type: Literal["standard", "vlm", "openai"] = "standard",
        save_artifacts: bool = False,
        image_mode: Literal["placeholder", "embedded", "referenced"] = "placeholder",
    ) -> None:
        self.pipeline_type = pipeline_type
        self.save_artifacts = save_artifacts
        self.image_mode = image_mode
        self.validate_image_mode()

        if pipeline_type == "standard":
            self.converter = self.build_standard_converter()
        elif pipeline_type == "vlm":
            self.ensure_vllm_server()
            self.converter = self.build_vlm_converter(
                runtime_type=VlmEngineType.API,
                url="http://localhost:8912/v1/chat/completions",
                params={
                    "model": "ibm-granite/granite-docling-258M",
                    "max_tokens": 4096,
                    "skip_special_tokens": True,
                },
            )
        elif pipeline_type == "openai":
            self.converter = self.build_vlm_converter(
                runtime_type=VlmEngineType.API_OPENAI,
            )
        else:
            raise ValueError(f"Invalid pipeline_type: {pipeline_type}")

    def validate_image_mode(self) -> None:
        if self.save_artifacts:
            valid_modes = {"embedded", "referenced"}
            if self.image_mode not in valid_modes:
                raise ValueError(
                    "Invalid image mode for saving artifacts; only 'embedded' and 'referenced' are supported."
                )
            return

        valid_modes = {"placeholder", "embedded"}
        if self.image_mode not in valid_modes:
            raise ValueError(
                "Invalid image mode for export; only 'placeholder' and 'embedded' are supported."
            )

    def build_standard_converter(
        self,
        force_cpu: bool = False,
    ) -> DocumentConverter:
        use_cuda = not force_cpu and torch.cuda.is_available()
        accelerator_options = AcceleratorOptions(
            device=(
                AcceleratorDevice.CUDA
                if use_cuda
                else AcceleratorDevice.CPU if force_cpu else "auto"
            )
        )

        if use_cuda:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(
                "Detected GPU: {} ({:.2f} GB VRAM)",
                torch.cuda.get_device_name(0),
                vram_gb,
            )
            if vram_gb > 24:
                batch_size = 128
            elif vram_gb >= 20:
                batch_size = 64
            else:
                batch_size = 16
        else:
            batch_size = 2
            if force_cpu:
                logger.warning("Using CPU fallback for PDF parsing")
            else:
                logger.info("No GPU detected; using CPU for parsing")

        pipeline_options = ThreadedPdfPipelineOptions(
            accelerator_options=accelerator_options,
            ocr_batch_size=batch_size,
            layout_batch_size=batch_size,
            table_batch_size=4,
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

    def should_retry_standard_parse_on_cpu(self, exc: Exception) -> bool:
        if self.pipeline_type != "standard":
            return False

        message = str(exc).lower()
        retry_keywords = (
            "cudnn",
            "libcublas",
            "cuda driver",
            "cuda runtime",
            "cuda error",
        )
        return any(keyword in message for keyword in retry_keywords)

    def ensure_vllm_server(self) -> None:
        logger.info("Checking whether the VLLM server is running")
        retries = 5
        while retries > 0:
            try:
                response = requests.get("http://localhost:8912/v1/models", timeout=2)
                if response.status_code != 200:
                    logger.warning(
                        "VLLM server responded with status {}", response.status_code
                    )
                    raise RuntimeError("VLLM server not responding correctly")
                return
            except requests.exceptions.RequestException:
                logger.warning("VLLM server is not running at http://localhost:8912")
                logger.warning(
                    "Run: vllm serve ibm-granite/granite-docling-258M --revision untied --host 127.0.0.1 --port 8912 --gpu-memory-utilization 0.9 --max-num-seqs 1024 --max-num-batched-tokens 16384 --enable-chunked-prefill"
                )
                retries -= 1
                time.sleep(5)

        raise RuntimeError("VLLM server not available after 5 attempts")

    def build_vlm_converter(
        self,
        *,
        runtime_type: VlmEngineType,
        url: str | None = None,
        params: dict | None = None,
    ) -> DocumentConverter:
        engine_options_kwargs = {
            "runtime_type": runtime_type,
            "timeout": 90,
        }
        if url is not None:
            engine_options_kwargs["url"] = url
        if params is not None:
            engine_options_kwargs["params"] = params

        engine_options = ApiVlmEngineOptions(**engine_options_kwargs)
        vlm_options = VlmConvertOptions.from_preset(
            "granite_docling",
            engine_options=engine_options,
        )
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                )
            }
        )

    def parse_single_pdf(
        self,
        pdf_path: str,
        format_type: Literal["md", "json", "html"] = "md",
        save_path: str | Path | None = None,
    ) -> str | dict:
        try:
            parsed_document = self.converter.convert(pdf_path)
        except Exception as exc:
            if not self.should_retry_standard_parse_on_cpu(exc):
                raise

            logger.warning(
                "PDF parsing hit a CUDA runtime error for {}; retrying on CPU: {}",
                pdf_path,
                exc,
            )
            self.converter = self.build_standard_converter(force_cpu=True)
            parsed_document = self.converter.convert(pdf_path)

        if save_path is not None:
            artifacts_dir = (
                self.get_artifacts_dir(save_path) if self.save_artifacts else None
            )
            if format_type == "md":
                parsed_document.document.save_as_markdown(
                    filename=save_path,
                    artifacts_dir=artifacts_dir,
                    image_mode=self.image_mode,
                )
                return ""
            if format_type == "json":
                parsed_document.document.save_as_json(
                    filename=save_path,
                    artifacts_dir=artifacts_dir,
                    image_mode=self.image_mode,
                )
                return ""
            if format_type == "html":
                parsed_document.document.save_as_html(
                    filename=save_path,
                    artifacts_dir=artifacts_dir,
                    image_mode=self.image_mode,
                )
                return ""
            raise ValueError(f"Invalid save format: {format_type}")

        if format_type == "md":
            return parsed_document.document.export_to_markdown(
                image_mode=self.image_mode
            )
        if format_type == "json":
            return parsed_document.document.export_to_dict(image_mode=self.image_mode)
        if format_type == "html":
            return parsed_document.document.export_to_html(image_mode=self.image_mode)
        raise ValueError(f"Invalid save format: {format_type}")

    def parse_multiple_pdfs(
        self,
        pdf_paths: list[str],
        format_type: Literal["md", "json", "html"] = "md",
        save_paths: list[str | Path] | None = None,
    ) -> list[str | dict]:
        try:
            converted_results = self.converter.convert_all(pdf_paths)
        except Exception as exc:
            if not self.should_retry_standard_parse_on_cpu(exc):
                raise

            logger.warning(
                "PDF batch parsing hit a CUDA runtime error; retrying on CPU: {}",
                exc,
            )
            self.converter = self.build_standard_converter(force_cpu=True)
            converted_results = self.converter.convert_all(pdf_paths)

        if save_paths is not None:
            for index, result in enumerate(converted_results):
                artifacts_dir = None
                if self.save_artifacts:
                    artifacts_dir = self.get_artifacts_dir(save_paths[index])
                    logger.info("Saving artifacts to {}", artifacts_dir)
                if format_type == "md":
                    result.document.save_as_markdown(
                        filename=save_paths[index],
                        artifacts_dir=artifacts_dir,
                        image_mode=self.image_mode,
                    )
                elif format_type == "json":
                    result.document.save_as_json(
                        filename=save_paths[index],
                        artifacts_dir=artifacts_dir,
                        image_mode=self.image_mode,
                    )
                elif format_type == "html":
                    result.document.save_as_html(
                        filename=save_paths[index],
                        artifacts_dir=artifacts_dir,
                        image_mode=self.image_mode,
                    )
                else:
                    raise ValueError(f"Invalid save format: {format_type}")
            return []

        if format_type == "md":
            return [
                result.document.export_to_markdown(image_mode=self.image_mode)
                for result in converted_results
            ]
        if format_type == "json":
            return [
                result.document.export_to_dict(image_mode=self.image_mode)
                for result in converted_results
            ]
        if format_type == "html":
            return [
                result.document.export_to_html(image_mode=self.image_mode)
                for result in converted_results
            ]
        raise ValueError(f"Invalid save format: {format_type}")

    def parse_pdf_directory(
        self,
        pdf_directory: str,
        format_type: Literal["md", "json", "html"] = "md",
        save_directory: str | Path | None = None,
    ) -> list[str | dict]:
        pdf_paths = [str(path) for path in Path(pdf_directory).glob("*.pdf")]
        if save_directory is None:
            return self.parse_multiple_pdfs(pdf_paths, format_type)

        os.makedirs(save_directory, exist_ok=True)
        save_paths = [
            os.path.join(
                save_directory,
                os.path.basename(path).replace(".pdf", f".{format_type}"),
            )
            for path in pdf_paths
        ]
        logger.info("Saving parsed files to {}", save_directory)
        self.parse_multiple_pdfs(pdf_paths, format_type, save_paths)
        return []

    def get_artifacts_dir(self, save_path: str | Path) -> Path:
        save_path = Path(save_path).absolute()
        artifacts_dir = save_path.with_suffix(f".{self.image_mode}_artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir


if __name__ == "__main__":
    from src.setup import bootstrap_runtime

    bootstrap_runtime()
    parser = PDFParser(
        pipeline_type="standard",
        save_artifacts=False,
        image_mode="placeholder",
    )
    parser.parse_pdf_directory(
        pdf_directory="resources/samples/acl",
        format_type="md",
        save_directory="./test_pdf_parser_output",
    )
