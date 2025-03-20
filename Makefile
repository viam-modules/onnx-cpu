MOD_ARCH := $(shell uname -m)
MOD_OS := $(shell uname -s)
test:
	go test
lint:
	golangci-lint run

module.tar.gz:
ifeq ($(VIAM_TARGET_OS),windows) # this needs to be at the top since windows is emulated
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 go build -a -o module.exe ./cmd/module
	jq '.entrypoint = "module.exe"' meta.json > temp.json && mv temp.json meta.json
	tar -czf $@ module.exe third_party/onnxruntime.dll meta.json
else ifeq ($(MOD_OS),Darwin)
ifeq ($(MOD_ARCH),x86_64)
	@echo "Unsupported OS: $(MOD_OS) or architecture: $(MOD_ARCH)"
else ifeq ($(MOD_ARCH),arm64)
	go build -a -o module ./cmd/module
	tar -czf $@ module third_party/onnxruntime_arm64.dylib
endif
else ifeq ($(MOD_OS),Linux)
ifeq ($(MOD_ARCH),x86_64)
	go build -a -o module ./cmd/module
	tar -czf $@ module third_party/onnxruntime.so
else ifeq ($(MOD_ARCH),arm64)
	go build -a -o module ./cmd/module
	tar -czf $@ module third_party/onnxruntime_arm64.so
else ifeq ($(MOD_ARCH),aarch64)
	go build -a -o module ./cmd/module
	tar -czf $@ module third_party/onnxruntime_arm64.so
endif
else
	@echo "Unsupported OS: $(MOD_OS) or architecture: $(MOD_ARCH)"
endif



module: NDK_ROOT ?= $(HOME)/Library/Android/sdk/ndk/26.1.10909125/
module: ARCH ?= aarch64
# note: this is 386 for x86 AVD -- retest amd64 on x86_64 AVD + update instructions
module: GOARCH ?= arm64
module: TARGET_API ?= android28
# onnx version basd on what onnxruntime_go targets
module: ONNX_VERSION ?= 1.16.1
module: SO_ARCH ?= arm64-v8a
module: CGO_ENABLED := 1
module: CC := $(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android30-clang
module: CGO_CFLAGS := -I$(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include \
							-I$(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include/aarch64-linux-android
module: CGO_LDFLAGS := -L$(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib
module:
	GOOS=android GOARCH=$(GOARCH) CGO_ENABLED=1 \
		CC=$(CC) \
		CGO_CFLAGS="$(CGO_CFLAGS)" \
		CGO_LDFLAGS="$(CGO_LDFLAGS)" \
		go build -v \
		-tags no_cgo \
		-o $@ \
		./cmd/module

onnxruntime-android-$(ONNX_VERSION).aar:
	# see https://onnxruntime.ai/docs/install/#install-on-android
	wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$(ONNX_VERSION)/onnxruntime-android-$(ONNX_VERSION).aar

third_party/onnx-android-$(SO_ARCH).so: onnxruntime-android-$(ONNX_VERSION).aar
	unzip -o $< 'jni/*/*.so'
	cp jni/$(SO_ARCH)/libonnxruntime.so $@

bundle-droid-$(SO_ARCH).tar.gz: module third_party/onnx-android-$(SO_ARCH).so
	tar -czf $@ $^
