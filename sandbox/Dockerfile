FROM mcr.microsoft.com/dotnet/sdk:10.0 AS development

# Add openssh-client to the installation layer
RUN apt-get update && apt-get install -y \
    mariadb-client \
    curl \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Standardize commands via scripts
RUN echo '#!/bin/bash\ndotnet build' > /usr/local/bin/agent-build && chmod +x /usr/local/bin/agent-build
RUN echo '#!/bin/bash\ndotnet test' > /usr/local/bin/agent-test && chmod +x /usr/local/bin/agent-test

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]