import grpc
from concurrent import futures
from loguru import logger

from app.config import settings
from app.grpc_diagnostics import set_grpc_listen
from app.logging_setup import configure_logging
from app.grpc_server.prediction_servicer import PredictionServicer


async def serve_grpc():
    configure_logging(settings.log_level)
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

    # Import generated code (run protoc first)
    try:
        from app.grpc_server.generated import prediction_service_pb2_grpc
        prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
            PredictionServicer(), server
        )
    except ImportError:
        logger.warning("gRPC stubs not generated yet. Run: make proto")
        set_grpc_listen("", False)
        return

    listen_addr = f"[::]:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)
    set_grpc_listen(listen_addr, True)
    logger.info(f"gRPC server listening on {listen_addr}")

    await server.start()
    await server.wait_for_termination()