#include "onnxruntime_wrapper.h"

static const OrtApi *ort_api = NULL;

int SetAPIFromBase(OrtApiBase *api_base) {
  if (!api_base) return 1;
  ort_api = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api) return 2;
  return 0;
}

void ReleaseOrtStatus(OrtStatus *status) {
  ort_api->ReleaseStatus(status);
}

OrtStatus *CreateOrtEnv(char *name, OrtEnv **env) {
  return ort_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, name, env);
}

OrtStatus *DisableTelemetry(OrtEnv *env) {
  return ort_api->DisableTelemetryEvents(env);
}

OrtStatus *EnableTelemetry(OrtEnv *env) {
  return ort_api->EnableTelemetryEvents(env);
}

void ReleaseOrtEnv(OrtEnv *env) {
  ort_api->ReleaseEnv(env);
}

OrtStatus *CreateOrtMemoryInfo(OrtMemoryInfo **mem_info) {
  return ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
    mem_info);
}

void ReleaseOrtMemoryInfo(OrtMemoryInfo *info) {
  ort_api->ReleaseMemoryInfo(info);
}

const char *GetErrorMessage(OrtStatus *status) {
  if (!status) return "No error (NULL status)";
  return ort_api->GetErrorMessage(status);
}

OrtStatus *CreateSession(void *model_data, size_t model_data_length,
  OrtEnv *env, OrtSession **out) {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  
  // TODO: enable optimizations in future
  //enum GraphOptimizationLevel opLevel= ORT_ENABLE_ALL;
  // status = ort_api->SetSessionGraphOptimizationLevel(options,opLevel );

  status = ort_api->CreateSessionOptions(&options);
  if (status) return status;
  status = ort_api->CreateSessionFromArray(env, model_data, model_data_length,
    options, out);
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *CreateSessionWithCUDA(void *model_data, size_t model_data_length,
  OrtEnv *env, OrtSession **out) {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;

  status = ort_api->CreateSessionOptions(&options);
  if (status) return status;
  
 // CUDA EP involves 4 steps; create->update->AppendTOsession ->Release(As defer)
 // step: create
  OrtCUDAProviderOptionsV2* cuda_options = NULL;
  status = ort_api->CreateCUDAProviderOptions(&cuda_options);
  if (status) return status;
  const char *keys = "device_id";
  const char * values = "0";

  // step update with devide_id:0
  status = ort_api->UpdateCUDAProviderOptions(cuda_options, &keys, &values, 1);
  if (status) return status;

  status = ort_api-> SessionOptionsAppendExecutionProvider_CUDA_V2(options,cuda_options);
  if (status) return status;
   // TODO: enable optimizations in future
  /* enum GraphOptimizationLevel opLevel= ORT_ENABLE_ALL;
  status = ort_api->SetSessionGraphOptimizationLevel(options,opLevel );
  if (status) return status;*/

  status = ort_api->CreateSessionFromArray(env, model_data, model_data_length,
    options, out);
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *RunOrtSession(OrtSession *session,
  OrtValue **inputs, char **input_names, int input_count,
  OrtValue **outputs, char **output_names, int output_count) {
  OrtStatus *status = NULL;
  status = ort_api->Run(session, NULL, (const char* const*) input_names,
    (const OrtValue* const*) inputs, input_count,
    (const char* const*) output_names, output_count, outputs);
  return status;
}

void ReleaseOrtSession(OrtSession *session) {
  ort_api->ReleaseSession(session);
}

void ReleaseOrtValue(OrtValue *value) {
  ort_api->ReleaseValue(value);
}

OrtStatus *CreateOrtTensorWithShape(void *data, size_t data_size,
  int64_t *shape, int64_t shape_size, OrtMemoryInfo *mem_info,
  ONNXTensorElementDataType dtype, OrtValue **out) {
  OrtStatus *status = NULL;
  status = ort_api->CreateTensorWithDataAsOrtValue(mem_info, data, data_size,
    shape, shape_size, dtype, out);
  return status;
}
